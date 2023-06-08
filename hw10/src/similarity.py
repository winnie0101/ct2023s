import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
import warnings

warnings.filterwarnings("ignore")

import glob
import os
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Step 3 compare your handwriting with others")

    parser.add_argument("--myId", help="Your student id", default="111C52034", type=str)

    parser.add_argument(
        "--targetId",
        help="Target student id, txt as target list or single id",
        type=str,
    )

    parser.add_argument(
        "--markDatabase", help="A txt file to save all compare result", type=str
    )

    parser.add_argument(
        "--maxCompare",
        help="Max number of word to compare with target",
        default=13589,
        type=int,
    )

    parser.add_argument(
        "--skipExist",
        help="Skip the word if exist in markDatabase",
        action="store_true",
    )

    parser.add_argument(
        "--crossCompare",
        help="Compare all student with each other, a txt file with studentID list,\
                              will ignore --myId, --targetId and --markDatabase",
        type=str,
    )

    args = parser.parse_args()
    return args


def getMarkDatabase(args: ArgumentParser) -> dict:
    """
    Read markDatabase based from argument setting, if not exist, return empty dict
    """
    if not args.markDatabase:
        args.markDatabase = f"{args.myId}_markDatabase.txt"

    if os.path.exists(args.markDatabase):
        with open(args.markDatabase, "r", encoding="UTF-8") as f:
            dataFile = f.readlines()
    else:
        return dict()

    markDatabase = {}
    for line in dataFile:
        target_ID, MSE, SSIM, LPIPS = line.strip().split()
        markDatabase[target_ID] = {
            "MSE": float(MSE),
            "SSIM": float(SSIM),
            "LPIPS": float(LPIPS),
        }
    return markDatabase


def dumpMarkDatabase(args: ArgumentParser, markDatabase: dict) -> None:
    """
    Dump markDatabase to txt file
    """
    with open(args.markDatabase, "w") as f:
        for target_ID in markDatabase:
            MSE = markDatabase[target_ID]["MSE"]
            SSIM = markDatabase[target_ID]["SSIM"]
            LPIPS = markDatabase[target_ID]["LPIPS"]
            f.write(f"{target_ID} {MSE} {SSIM} {LPIPS}\n")


def compareWithTarget(
    args: ArgumentParser,
    target_ID: str,
    word_list: list,
    markDatabase: dict,
    loss_fn,
    device,
) -> None:
    success = total_LPIPS = total_MSE = total_SSIM = 0
    pbar = tqdm(word_list, desc="Calculating similarity")
    for i, word_path in enumerate(pbar):
        try:
            _, filename = os.path.split(word_path)

            # Check file exist
            if not (
                os.path.exists(f"./1_138_{args.myId}/{filename}")
                and os.path.exists(f"./1_138_{target_ID}/{filename}")
            ):
                raise FileNotFoundError

            # 自己的手寫字圖片路徑
            myWordImg = cv2.imread(
                f"./1_138_{args.myId}/{filename}", cv2.IMREAD_GRAYSCALE
            )
            myWord = torch.from_numpy(myWordImg).unsqueeze(0).unsqueeze(0).float() / 255.0
            myWord = myWord.to(device)

            # 別人的手寫字圖片路徑
            targetWordImg = cv2.imread(
                f"./1_138_{target_ID}/{filename}", cv2.IMREAD_GRAYSCALE
            )
            targetWord = (
                torch.from_numpy(targetWordImg).unsqueeze(0).unsqueeze(0).float() / 255.0
            )
            targetWord = targetWord.to(device)

            # 計算分數
            mse = np.mean((myWordImg - targetWordImg) ** 2)  # 計算MSE
            ssim_score = ssim(myWordImg, targetWordImg, win_size=7)  # 計算SSIM相似度
            lpips_distance = loss_fn(myWord, targetWord)  # 計算LPIPS距離

            # Accumulate the scores
            success += 1
            total_MSE += mse
            total_SSIM += ssim_score
            total_LPIPS += lpips_distance.item()

            if i >= args.maxCompare:
                break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error occured at {word_path}, {e}")
            exit()

    if success != 0:
        if target_ID not in markDatabase:
            markDatabase[target_ID] = dict()
        markDatabase[target_ID]["MSE"] = total_MSE / success
        markDatabase[target_ID]["SSIM"] = total_SSIM / success
        markDatabase[target_ID]["LPIPS"] = total_LPIPS / success
        print(
            f'Target: {target_ID} \
            MSE: {markDatabase[target_ID]["MSE"]:.5f} \
            SSIM: {markDatabase[target_ID]["SSIM"]:.5f} \
            LPIPS: {markDatabase[target_ID]["LPIPS"]:.5f}'
        )


def printMostSimilar(args: ArgumentParser, markDatabase: dict) -> None:
    """
    Print the most similar comparison result
    """
    minMSE_ID = min(markDatabase, key=lambda x: markDatabase[x]["MSE"])
    maxSSIM_ID = max(markDatabase, key=lambda x: markDatabase[x]["SSIM"])
    minLPIPS_ID = min(markDatabase, key=lambda x: markDatabase[x]["LPIPS"])

    print(f"Compare ID: {args.myId}")
    print(f'Most similar by MSE: {minMSE_ID} {markDatabase[minMSE_ID]["MSE"]:.5f}')
    print(f'Most similar by SSIM: {maxSSIM_ID} {markDatabase[maxSSIM_ID]["SSIM"]:.5f}')
    print(
        f'Most similar by LPIPS: {minLPIPS_ID} {markDatabase[minLPIPS_ID]["LPIPS"]:.5f}'
    )


def main(args):
    word_list = glob.glob(f"1_138_{args.myId}/*.png")
    markDatabase = getMarkDatabase(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入預訓練的LPIPS模型
    loss_fn = lpips.LPIPS(net="alex").to(device)

    if args.targetId.endswith(".txt"):
        # 讀取目標名單（學號）
        with open(args.targetId, "r") as f:
            targets = f.readlines()
    else:
        targets = [args.targetId]

    # 逐一計算與“args.myId"的相似度
    for target in targets:
        target_ID = target.strip().split()[0]
        # 跳過已經計算過的和自己
        if (args.skipExist and target_ID in markDatabase) or target_ID == args.myId:
            continue
        compareWithTarget(args, target_ID, word_list, markDatabase, loss_fn, device)
        # 保存結果
        dumpMarkDatabase(args, markDatabase)

    # 輸出成績
    printMostSimilar(args, markDatabase)


if __name__ == "__main__":
    args = parse_args()

    if args.crossCompare:
        # 幫全名單的人計算他們與自己最相似的字體
        with open(args.crossCompare, "r", encoding="UTF-8") as f:
            targets = f.readlines()

        for target in targets:
            target_ID, _ = target.strip().split()
            args.myId = target_ID
            args.markDatabase = f"{args.myId}_markDatabase.txt"
            args.targetId = args.crossCompare
            main(args)
    else:
        main(args)

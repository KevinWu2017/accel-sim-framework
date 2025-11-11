#  从远程服务器下载 GPU 应用程序的 trace 文件（压缩包），供 Accel-Sim 模拟使用。
#!/usr/bin/env python3

from optparse import OptionParser
import re
import os
import math
import sys
from subprocess import Popen, STDOUT

VERSION = "1.1.0"
WEB_DIRECTORY = "ftp://ftp.ecn.purdue.edu/tgrogers/accel-sim/traces/"
this_directory = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.insert(0, os.path.join(this_directory, "util", "job_launching"))
import common

millnames = ["", " K", " M", " G", " T"]

# 将带单位的字符串转为数字
def getNumRaw(n):
    try:
        return float(n)
    except ValueError:
        count = 0
        for name in millnames:
            if n[-1].strip() == name.strip():
                n = float(n[:-1].strip()) * 10 ** (3 * count)
                break
            count += 1
        return float(n)

# 将数字格式化为带单位的字符串
def millify(n):
    n = getNumRaw(n)
    if math.isnan(n):
        return "NaN"
    if math.isinf(n):
        return "inf"
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )
    return "{:.2f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


# 表示一个应用程序套件（如 rodinia-3.1）
class Suite:
    def __init__(self, name):
        self.name = name
        self.uncompressedSize = None
        self.compressedSize = None


# 表示一种 GPU 型号（如 tesla-v100）
class Card:
    def __init__(self, name):
        self.name = name
        self.suites = {}   # 包含多个 Suite

    def getTotalCompressed(self):    # 所有压缩包总大小
        total = 0.0
        for name, suite in self.suites.items():
            try:
                total += suite.compressedSize
            except TypeError:
                sys.exit("Problem with compressed size in suite {0}".format(name))
        return total

    def getTotalUncompressed(self):   # 所有解压后总大小
        total = 0.0
        for name, suite in self.suites.items():
            total += suite.uncompressedSize
        return total

# 下载单个 trace
def downloadTrace(cardName, suiteName):
    webFile = os.path.join(
        WEB_DIRECTORY, cardName, VERSION + ".latest", suiteName + ".tgz"
    )
    print("\n\nDownloading {0}".format(webFile))
    wget = Popen(["wget " + webFile], stderr=STDOUT, shell=True)
    wget.communicate()
    if wget.returncode != 0:
        sys.exit("wget {0} returned {1}".format(webFile, wget.returncode))


def main():
    # 解析命令行参数
    parser = OptionParser()
    parser.add_option(
        "-a",
        "--apps",
        dest="apps",
        default=None,
        help="Pass the comma seperated input list instead of asking "
        "from stdin. Example: -a tesla-v100/rodinia-3.1,tesla-v100/cudasdk",
    )
    parser.add_option(
        "-d",
        "--download_dir",
        dest="download_dir",
        default="",
        help="Directory to download the traces to.",
    )
    (options, args) = parser.parse_args()
    
    # 设置下载目录
    try:
        hw_run_dir = common.dir_option_test(
            options.download_dir, "hw_run", this_directory
        )
    except common.PathMissing:
        hw_run_dir = os.path.join(this_directory, "hw_run")

    if not os.path.exists(hw_run_dir):
        os.makedirs(hw_run_dir)
    os.chdir(hw_run_dir)
    
    # 下载 trace 索引文件
    # Parse the trace summary
    trace_summary = os.path.join(hw_run_dir, VERSION + ".trace.summary.txt")
    try:
        os.remove(trace_summary)
    except OSError:
        pass

    Popen(
        ["wget " + " " + WEB_DIRECTORY + VERSION + ".trace.summary.txt"],
        stderr=STDOUT,
        shell=True,
    ).communicate()

    #  解析索引文件
    lineFormat = re.compile(r"(.*)\t(.*)/" + VERSION + ".latest/(.*)")
    sizeDict = {}
    for line in open(trace_summary):
        lineMatch = lineFormat.match(line)
        if lineMatch != None:
            size = lineMatch.group(1)
            cardName = lineMatch.group(2)
            fileName = lineMatch.group(3)
            suiteName = re.sub(r"(.*)\.tgz", r"\1", fileName)
            if cardName not in sizeDict:
                sizeDict[cardName] = Card(cardName)
            card = sizeDict[cardName]
            if suiteName not in card.suites:
                newSuite = Suite(suiteName)
                card.suites[suiteName] = Suite(suiteName)

            suite = sizeDict[cardName].suites[suiteName]
            if suiteName != fileName:
                suite.compressedSize = getNumRaw(size)
            else:
                suite.uncompressedSize = getNumRaw(size)

    # 显示可用 trace 信息
    # Infor the user what is available - ask them what they want to do
    print("\n\nCurrently Available Traces:")
    for cardName, card in sizeDict.items():
        print(
            "GPU Name: {0}. All Apps Compressed (Download size): {1}, All Apps Uncompressed (Size on disk when used): {2}".format(
                card.name,
                millify(card.getTotalCompressed()),
                millify(card.getTotalUncompressed()),
            )
        )
        for name, suite in card.suites.items():
            print(
                "\t{0}: Compressed = {1}, Uncompressed = {2}".format(
                    suite.name,
                    millify(suite.compressedSize),
                    millify(suite.uncompressedSize),
                )
            )

    # 用户选择要下载的内容
    selectionValid = False
    while not selectionValid:
        if options.apps == None:
            selection = input(
                "\n-------\nWhat do you want to download?"
                "\n<card/suite>,<card/suite> (i.e. tesla-v100/rodinia-3.1,tesla-v100/cudasdk)"
                "\n(Default=all/all) : "
            )
            if selection == "" or selection == None:
                selection = "all/all"
        else:
            selection = options.apps
        try:
            for item in selection.split(","):
                cardName = item.split(r"/")[0]
                suiteName = item.split(r"/")[1]
                if cardName == "all":
                    for cardName, card in sizeDict.items():
                        if suiteName == "all":
                            for suiteName, suite in card.suites.items():
                                downloadTrace(cardName, suiteName)
                        else:
                            downloadTrace(cardName, suiteName)
                else:
                    card = sizeDict[cardName]
                    if suiteName == "all":
                        for suiteName, suite in card.suites.items():
                            downloadTrace(cardName, suiteName)
                    else:
                        downloadTrace(cardName, suiteName)

            selectionValid = True
        except Exception as e:
            selectionValid = False
            print("Invalid Input: {0}".format(e))
            if options.apps != None:
                sys.exit(1)

    print(
        "\n\nDownload successful to {0}.\nFiles must be uncompressed with tar -xzvf <filename> to be usable by accel-sim".format(
            hw_run_dir
        )
    )


if __name__ == "__main__":
    main()

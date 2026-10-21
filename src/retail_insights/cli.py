"""Command-line entry points for independently verifiable modules."""

import argparse
import json

from .preparation import prepare_file


def main():
    parser = argparse.ArgumentParser(description='电商用户分析：数据导入与质量检查')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='导入 Excel/CSV 并生成清洗质量报告')
    prepare.add_argument('--input', required=True, help='原始 Excel/CSV 路径')
    prepare.add_argument('--root', help='产物根目录，默认项目目录')
    prepare.add_argument('--keep-duplicates', action='store_true', help='保留完全重复行')
    args = parser.parse_args()
    try:
        result = prepare_file(args.input, root=args.root, drop_duplicates=not args.keep_duplicates)
    except (ValueError, OSError) as error:
        parser.exit(1, f'导入失败：{error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

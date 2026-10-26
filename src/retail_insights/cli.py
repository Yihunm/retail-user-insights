"""Command-line entry points for independently verifiable modules."""

import argparse
import json

from pathlib import Path

from .preparation import prepare_file, load_dataset, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description='电商用户分析：数据导入、时间样本与预测训练')
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='导入 Excel/CSV 并生成清洗质量报告')
    prepare.add_argument('--input', required=True, help='原始 Excel/CSV 路径')
    prepare.add_argument('--root', help='产物根目录，默认项目目录')
    prepare.add_argument('--keep-duplicates', action='store_true', help='保留完全重复行')
    for command, help_text in [('build-samples', '构建完整时间窗口的带标签样本'),
                                ('train', '训练并比较模型与不平衡处理方案')]:
        subparser = commands.add_parser(command, help=help_text)
        subparser.add_argument('--task', required=True, choices=['purchase', 'churn'])
        subparser.add_argument('--root', help='项目数据根目录')
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare_file(args.input, root=args.root, drop_duplicates=not args.keep_duplicates)
        else:
            from .prediction_features import make_samples
            from .prediction_presentation import run_prediction, sample_summary
            root = Path(args.root or PROJECT_ROOT).resolve()
            data = load_dataset(root)
            if args.command == 'build-samples':
                samples = make_samples(data['sales'], task=args.task)
                destination = root / 'data' / 'processed' / data['metadata']['dataset_id']
                samples.to_parquet(destination / f'prediction_samples_{args.task}.parquet', index=False)
                result = {'rows': len(samples), 'metadata': samples.attrs,
                          'summary': sample_summary(samples).to_dict('records')}
            else:
                _, context = run_prediction(data['sales'], args.task, root, data['metadata']['dataset_id'])
                result = {key: context[key] for key in ['run_id', 'task', 'best_model', 'selected_strategy',
                           'threshold', 'validation_scores', 'test_scores', 'artifact_dir', 'elapsed_seconds']}
    except (ValueError, OSError) as error:
        parser.exit(1, f'操作失败：{error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

import argparse
from genrec.utils2 import parse_command_line_args, get_pipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='TIGER', help='Model name')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2014', help='Dataset name')
    parser.add_argument('--process_only', action='store_true', help='Only process data, do not train')
    parser.add_argument('--mode', type=str, default=None, help='Run mode: train or test')
    return parser.parse_known_args()



if __name__ == '__main__':

    args, unparsed_args = parse_args()

    command_line_configs = parse_command_line_args(unparsed_args)
    if args.mode is not None:
        command_line_configs['mode'] = args.mode

    pipeline = get_pipeline(args.model)      \
            (
        model_name=args.model,
        dataset_name=args.dataset,
        config_dict=command_line_configs
    )

    # Only run training if --process_only is not set
    if not args.process_only:
        pipeline.run()
    else:
        print(f"\n{'='*60}")
        print(f"✅ Data processing completed successfully!")
        print(f"   Dataset: {args.dataset}")
        print(f"   Model: {args.model}")
        print(f"   Processed data location: cache/{args.dataset}/")
        print(f"{'='*60}\n")

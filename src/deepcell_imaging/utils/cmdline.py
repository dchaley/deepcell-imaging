import argparse

from deepcell_imaging.gcp_batch_jobs import get_batch_indexed_task


def get_task_arguments(task_name, args_cls):
    parser = argparse.ArgumentParser(task_name, add_help=False)

    parser.add_argument(
        "--tasks_spec_uri",
        help="URI to a JSON file containing a list of task parameters. Specify this, OR, the individual parameters.",
        type=str,
        required=False,
    )

    # If needed, we could change this to act on an args list passed
    # as a parameter, instead of just using implicit sys.argv
    parsed_args, args_remainder = parser.parse_known_args()

    if parsed_args.tasks_spec_uri:
        if len(args_remainder) > 0:
            raise ValueError("Either pass --tasks_spec_uri alone, or not at all")

        return get_batch_indexed_task(parsed_args.tasks_spec_uri, args_cls)
    else:
        model_fields = args_cls.model_fields

        parser = argparse.ArgumentParser(task_name, parents=[parser])

        for key in model_fields.keys():
            field = model_fields[key]
            parser.add_argument(
                f"--{key}",
                help=field.description,
                type=field.annotation,
                required=field.is_required(),
            )

        parsed_args = parser.parse_args(args_remainder)
        kwargs = {k: v for k, v in vars(parsed_args).items() if v is not None}
        return args_cls(**kwargs)

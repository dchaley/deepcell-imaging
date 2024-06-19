from itertools import groupby
import logging
import platform
import re
import requests
import resource
import traceback


def get_gce_instance_type():
    # Call the metadata server.
    try:
        metadata_server = "http://metadata/computeMetadata/v1/instance/machine-type"
        metadata_flavor = {"Metadata-Flavor": "Google"}

        # This comes back like this: projects/1234567890/machineTypes/n2-standard-8
        full_machine_type = requests.get(metadata_server, headers=metadata_flavor).text
        return full_machine_type.split("/")[-1]
    except Exception as e:
        exception_string = traceback.format_exc()
        logging.warning("Error getting machine type: " + exception_string)
        return "error"


def get_gce_region():
    # Call the metadata server.
    try:
        metadata_server = "http://metadata/computeMetadata/v1/instance/zone"
        metadata_flavor = {"Metadata-Flavor": "Google"}

        # This comes back like this: projects/projectnumber/zones/europe-west4-b
        region = requests.get(metadata_server, headers=metadata_flavor).text
        m = re.search("zones/(.*)-[^-]*$", region)
        return m.group(1)
    except Exception as e:
        exception_string = traceback.format_exc()
        logging.warning("Error getting region: " + exception_string)
        return "error"


def get_gce_is_preemptible():
    # Call the metadata server.
    try:
        metadata_server = "http://metadata/computeMetadata/v1/instance/scheduling/preemptible"
        metadata_flavor = {"Metadata-Flavor": "Google"}

        # This comes back like this: projects/1234567890/machineTypes/n2-standard-8
        preemptible_str = requests.get(metadata_server, headers=metadata_flavor).text
        return preemptible_str.lower() == "true"
    except Exception as e:
        exception_string = traceback.format_exc()
        logging.warning("Error getting preemptible, assuming false. Error: " + exception_string)
        return False


def get_gpu_info():
    # Import here, to avoid the cost if not needed when loading the module.
    import tensorflow as tf

    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    gpu_details = [tf.config.experimental.get_device_details(gpu) for gpu in gpu_devices]

    gpus_by_name = {
        k: list(v) for k, v in groupby(gpu_details, key=lambda x: x["device_name"])
    }

    gpu_names = list(gpus_by_name.keys())

    if len(gpu_names) == 0:
        return "", 0
    elif len(gpu_names) == 1:
        return gpu_names[0], len(gpus_by_name[gpu_names[0]])
    else:
        raise "Dunno how to handle multiple gpu types"


def get_peak_memory():
    # The getrusage call returns different units on mac & linux.
    # Get the OS type from the platform library,
    # then set the memory unit factor accordingly.
    os_type = platform.system()
    # This is crude and impartial– but it works across my mac & Google Cloud
    if "Darwin" == os_type:
        memory_unit_factor = 1000000000
    elif "Linux" == os_type:
        memory_unit_factor = 1000000
    else:
        # Assume kb like linux
        logging.warning("Couldn't infer machine type from %s", os_type)
        memory_unit_factor = 1000000

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / memory_unit_factor

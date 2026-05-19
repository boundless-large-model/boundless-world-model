def pack_paths(data, *key_groups):
    """
    Example:
        pack_paths(dataset.data, ("video", "start_frame", "end_frame"))

        # before:
        # {"video": "a.mp4", "start_frame": 0, "end_frame": 16}

        # after:
        # {"video": "a.mp4", "start_frame": 0, "end_frame": 16,
        #  "video_start_frame_end_frame": {"video": "a.mp4", "start_frame": 0, "end_frame": 16}}
    """
    for sample in data:
        for group in key_groups:
            combined_key = "_".join(group)
            sample[combined_key] = {k: sample.get(k) for k in group}
    return data

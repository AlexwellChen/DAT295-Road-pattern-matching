# DAT295-Road-pattern-matching
### Data Preparation

First use `ColorExtractor/stitch.py` generate target area images for SPQ.

Prepar `ved-final.zip` from  https://chalmers-my.sharepoint.com/:u:/g/personal/chenfa_chalmers_se/EX3bb1odQqNHq7snaoty4QEBn5UBpt1jZ1h-vVREjIO6Zg?e=2eb6o1 (Need Chalmers account to download file). Unzip and place in the root directory.

Prepar your GMAPS_API in `ValidateTool/tool.py` for GRQ.

### SPQ usage

The code for SPQ is in the `ColorExtractor/tools.py` file, and the corresponding function is `image_process_position_seq()`. This function contains the following parameters.

* lng_seq: trajectory coordinate longitude sequence
* lat_seq: latitude sequence of track coordinates
* idx: index of Input
* rest_point_delta: the difference of the remaining points in the sequence
* max_delta: the maximum value of the difference.
* flag: Print image flag 

Where `rest_point_delta` and `max_delta` are generated by `tools.point_delta()`, and the input of this function is the three variables `lng_seq` `lat_seq` `idx`.

You can refer to the code in `ColorExtractor/SPQ_Accuracy.py` for details on how to use it.

### GRQ usage

Main part of GRQ is in `ValidateTool/GRQ.py`, input a Road object and image name. 

Road object is generated by Google Map scraper, check `ValidateTool/tool.py` for details on how to use it (line 824).

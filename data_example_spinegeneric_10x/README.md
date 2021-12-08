# data_example_spinegeneric
Dataset extracted from [SpineGeneric](https://github.com/spine-generic/data-multi-subject). For each subject, all contrasts are registered on the GRE-T1w contrast.

### About this dataset

This dataset has been generated by:
- Randomly picking ten subjects from [SpineGeneric](https://github.com/spine-generic/data-multi-subject).
- Processing anatomical data using this [processing pipeline](https://github.com/ivadomed/ivadomed/tree/master/dev/prepare_data) which, for each subject, automatically segment the spinal cord on the GRE-T1w volume and registered all volumes to the GRE-T1w volume.
- Manually correcting the spinal cord masks.

### Download zip package (recommended)

If you are only planning on using the dataset for processing, you can download the latest version as a zip package:

~~~
curl -o ivadomed_spinegeneric_registered.zip -L https://github.com/ivadomed/data_spinegeneric_registered/archive/master.zip
unzip ivadomed_spinegeneric_registered.zip
~~~

### Clone the repository (slower)

If you are planning on contributing to this repository (e.g. uploading manual segmentations/labels), you need to clone this repository:
~~~
git clone https://github.com/ivadomed/data_spinegeneric_registered.git
~~~
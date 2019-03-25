from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets


class BIDSSegPair2D(mt_datasets.SegmentationPair2D):
    def __init__(self, input_filename, gt_filename, metadata):
        super().__init__(input_filename, gt_filename)
        self.metadata = metadata

    def get_pair_slice(self, slice_index, slice_axis=2):
        dreturn = super().get_pair_slice(slice_index, slice_axis)
        dreturn["input_metadata"]["bids_metadata"] = self.metadata
        return dreturn


class MRI2DBidsSegDataset(mt_datasets.MRI2DSegmentationDataset):
    def _load_filenames(self):
        for input_filename, gt_filename, bids_metadata in self.filename_pairs:
            segpair = BIDSSegPair2D(input_filename, gt_filename,
                                    bids_metadata)
            self.handlers.append(segpair)


class BidsDataset(MRI2DBidsSegDataset):
    def __init__(self, root_dir, slice_axis=2, cache=True,
                 transform=None, slice_filter_fn=None,
                 canonical=False, labeled=True,
                 normalize_metadata=False):
        self.bids_ds = bids.BIDS(root_dir)
        self.filename_pairs = []

        for subject in self.bids_ds.get_subjects():
            if not subject.has_derivative("labels"):
                print("Subject without derivative, skipping.")
                continue
            derivatives = subject.get_derivatives("labels")
            cord_label_filename = None
            for deriv in derivatives:
                if deriv.endswith("seg-manual.nii.gz"):
                    cord_label_filename = deriv
            if cord_label_filename is None:
                print("Subject without cord label.")
                continue

            if not subject.has_metadata():
                print("Subject without metadata.")
                continue

            metadata = subject.metadata()
            if "FlipAngle" not in metadata:
                print("{} without FlipAngle, skipping.".format(subject))
                continue
            elif "FlipAngle" in metadata and normalize_metadata:
            	metadata["FlipAngle"] = rescale_value(value_in=metadata["FlipAngle"],
            											range_in=[0.0, 360.0],
            											range_out=[0.0, 90.0])

            if "EchoTime" not in metadata:
                print("{} without EchoTime, skipping.".format(subject))
                continue

            if "RepetitionTime" not in metadata:
                print("{} without RepetitionTime, skipping.".format(subject))
                continue

            self.filename_pairs.append((subject.record.absolute_path,
                                        cord_label_filename, metadata))

        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical, normalize_metadata)

    def rescale_value(value_in, range_in, range_out):
        delta_in = range_in[1] - range_in[0]
        delta_out = range_out[1] - range_out[0]
        return (delta_out * (value_in - range_in[0]) / delta_in) + range_out[0]

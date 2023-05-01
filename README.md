# PigAggression

For training: 
  1. Pre-processor training
  2. Trainer
 
For testing/predicting:
  1. Pre-processor testing
  2. Video analysis

There are 2 pre-processors because the training version saves small (resized) versions of frames, while the testing version saves full size. 
Full size is required for clear video analysis output. It does not re-extract videos and frames as they are already saved by the training version.
When using new data, both pre-processors must be run for predicting (though re-training is recommended anyway).

Everything is unified for 1 smooth process in the PigAggressionRecognitionTool

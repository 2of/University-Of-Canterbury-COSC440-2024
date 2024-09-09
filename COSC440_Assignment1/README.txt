# Answer the conceptual questions here
Q1: Is there anything we need to know to get your code to work? If you did not get your code working or handed in an incomplete solution please let us know what you did complete (0-4 sentences)

run ./assignment.py as needed. 
./assignment_tests.py as needed.
MNIST in ./MNIST_data

Q2: Why do we normalize our pixel values between 0-1? (1-3 sentences)

Bounding between zero and one is slightly arbitrary (we're not doing anything with activation funcs who would be troubled here) but it is still prudent to bound to some consistent interval. Doing so mitigats blow outs from outliers. (betters numerical stability). Also MNIST is 
an 'analogue' set of data (as analogue as greyscale images get) so it removes extreme values that could occur given this. 

Q3: Why do we use a bias vector in our forward pass? (1-3 sentences)

Bias vector ensures that even if the wx calculation for some weights yields zero; that still no perceptron only outputs zeroes. This helps with ensuring our decision boundary actually moves on zeroed results. 


Q4: Why do we separate the functions for the gradient descent update from the calculation of the gradient in back propagation? (2-4 sentences)

Firstly; modularity is always a good thing with steps like this in terms of implementation. It certainly helps with debugging and test cases. We could substitute in and out different optimization (instead of backprop) methods too 
or particulary; extend the model such that more than one is used.
Secondly: (slightly dependant on the implementation), we can leverage this separation in the context of parallelization; because gradients & backprop are in batches, we could still be computing 
some of func A while func B begins; without the need for func A to fully complete. (some vagueness as we don't know precisely how this would work on each gpu etc) but this best utilizes our resources.


Q5: What are some qualities of MNIST that make it a “good” dataset for a classification problem? (2-3 sentences)

MNIST is convieniently 60k train and 10k test which are convieniently already labelled. The data is consistent 28x28 and it is also small enough to fit into very small memory and consistent!. Persumably; the breadth of images captures all variations of each character, so it
holds that this is a good dataset with which to represent unseen data very well. Aside from normalization, there is no signgifcant preprocessing required. Conversely, unseen data will require preprocessing to make it 'look' like the MNIST data. (as in greyscale, thresholded images)



Q6: Suppose you are an administrator of the NZ Health Service (CDHB or similar). What positive and/or negative effects would result from deploying an MNIST-trained neural network to recognize numerical codes on forms that are completed by hand by a patient when arriving for a health service appointment? (2-4 sentences)

Positive: Increased speediness / efficiency of new papers. Hand written forms are still variable, MNIST doesn't seem to contain characters who link to other characters, people who write their DOB
for example often link '1998' altogether. However, for things that are not hugely important if false positive, like ordering snacks for the break room, there can be a huge efficiency increase.

Negatives: Our model specifically achieves only ~85.9% accuracy.  The consequence of false positives and false negatives are both quite high. We could mitigate this somewhat by having a confidence minimum on the 
final probabilities; i.e. there is no detection if the mdoel is not 85% sure that this is the argmax class. The extreme case should be checked by a human (i.e. we do not want a patient filling out their patient number (?) 
to find that the number of the person they wrote vs the detected number doesnt match, presumably, very crucial numbers are checked by two staff, so the model could replace one.)


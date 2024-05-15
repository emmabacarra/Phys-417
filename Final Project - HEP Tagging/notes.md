the dataset contains information about particle collision events for a jet and the constituents. one file corresponds to a single final state particle that results from a decaying particle (debris from a particle collision), where it is essentially a matrix of a single row and two columns:

the element in the left column 'x' is a matrix that represents the features of the event. in this matrix, there are 5 columns that correspond to the following (in order from left to right): momentum, angular coordinate eta, angular coordinate phi, energy, and distance from the center of the jet. each row in this matrix corresponds to a constituent in the collision, which is variable in number between final state particles. the order of the constituents is not relevant or based on a time sequence, so the rows can be shuffled.

the element in the right column 'y' is a matrix of 5 rows and 1 column that represents the classification of the final state particle. each row represents the possible classification types in this order (from top to bottom): gluon, light quark, W boson, Z boson, top quark. the type of the final state particle is indicated by a 0 or 1 for each row, where 0 means false and 1 means true. for example, if the 'x' column element corresponds to a matrix of event properties that result in a light quark as the final state particle, the 'y' column element will contain a matrix that looks like this: [[0], [1], [0], [0], [0]].

the goal is to design a neural network with pytorch that can train with the event feature/property matrices (the 'x' columns) of several files (noting that between files, the matrices can vary in row number), evaluate its training with the corresponding 'y' columns, and use that training to predict the classification types of final state particles from additional, unseen files. 

------------------------------------

so far, i've experimented with trying a transformer model without masking and preprocessing the data with sklearn's standard scaler. the architecture includes an encoder layer but not a decoder layer, and with training i've only been able to get around 40% accuracy. 

ideas for preprocessing the raw data (noting that the event features can vary in row dimension), 
and the best approach to designing the right architecture and picking the proper techniques to use?

preprocessing
- scaling: try IQR scaler (interquartile range)
- definitely do padding to match max dimension
- feature selection techniques like LASSO regression or permutation importance

architecture
- try LSTM or GRUs? these are capable of handling variable-length sequences
- 1D CNN layer inside an RNN to help extract relevant spatial patterns
- adding an attention mechanism
- adding class weights or over/under sampling to account for imbalances in types

padding with LSTMs
- Traditional LSTMs: These require fixed-length inputs. In your case, since the 'x' matrices have a variable number of rows, padding with a specific value (e.g., zeros) is typically necessary. This ensures all sequences have the same length for processing by the LSTM layer.
- Packed LSTMs: PyTorch offers functionalities for Packed LSTMs, which allow you to handle sequences of different lengths without padding. This approach keeps track of the actual sequence lengths for each input and utilizes this information during training.

masking (if adding pads to data)
1. Pad the 'x' matrices with 0 representing "no information."
2. Create a mask tensor with the same dimensions as the padded 'x' matrices. Fill the mask tensor with 1s for non-padded elements and 0s for padded elements.
3. Use a masked LSTM layer that takes the 'x' matrices and the mask tensor as input. The mask will prevent the LSTM from processing padded elements during backpropagation.

combining attention mechanism with masking
1. Pad your 'x' matrices with a specific value (e.g., -1) representing "no information."
2. Create a mask tensor with the same dimensions as the padded 'x' matrices. Fill the mask tensor with 1s for non-padded elements and 0s for padded elements.
3. Use an attention layer within your RNN architecture (e.g., LSTM). This layer takes the padded 'x' matrix and the mask tensor as input.
4. The attention mechanism calculates attention scores for each element in the 'x' matrix. However, during these calculations, the mask is applied. It essentially multiplies the attention scores with the corresponding elements in the mask tensor. This ensures that scores for padded elements (with 0s in the mask) become zero, effectively removing their influence on the final attention weights.

packed LSTMs
- Packed Sequences Store Length Information: When you create a packed sequence using torch.nn.utils.rnn.pack_padded_sequence, it stores the actual length of each sequence within the batch. This information is crucial for the LSTM to process sequences of different lengths effectively.
- Padding Adds Unnecessary Information: Padding adds artificial elements (e.g., zeros) to shorter sequences to create a uniform length. This information is irrelevant to the actual particle collision event and can even hinder the LSTM's learning process.
- Packed Sequences Improve Efficiency: By avoiding padding, packed sequences utilize memory more efficiently, especially when dealing with sequences with significant variation in length.

using packed sequences:
1. Preprocess your data: Perform scaling/normalization on the features within the 'x' matrices.
2. Pad the sequences (optional): While not necessary for packed sequences, you can optionally pad all sequences to a maximum length for easier data handling before packing. However, this might not be the most efficient approach.
3. Sort the sequences (optional): Sorting the sequences by descending length can improve performance in some cases.
4. Create a packed sequence: Use torch.nn.utils.rnn.pack_padded_sequence to create a packed sequence object. This object will contain the actual sequence data and the corresponding sequence lengths.
5. Pass the packed sequence to the LSTM: Your LSTM layer can directly process the packed sequence object, taking into account the variable lengths of the sequences within the batch.
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Sigmoid function.
    This function accepts any shape of np.ndarray object as input and perform sigmoid operation.
    """
    return 1 / (1 + np.exp(-x))


def der_sigmoid(y):
    """ First derivative of Sigmoid function.
    The input to this function should be the value that output from sigmoid function.
    """
    return sigmoid(y) * (1 - sigmoid(y))


class GenData:
    @staticmethod
    def _gen_linear(n=100):
        """ Data generation (Linear)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data = np.random.uniform(0, 1, (n, 2))

        inputs = []
        labels = []

        for point in data:
            inputs.append([point[0], point[1]])

            if point[0] > point[1]:
                labels.append(0)
            else:
                labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def _gen_xor(n=100):
        """ Data generation (XOR)

        Args:
            n (int):    the number of data points generated in total.

        Returns:
            data (np.ndarray, np.float):    the generated data with shape (n, 2). Each row represents
                a data point in 2d space.
            labels (np.ndarray, np.int):    the labels that correspond to the data with shape (n, 1).
                Each row represents a corresponding label (0 or 1).
        """
        data_x = np.linspace(0, 1, n // 2)

        inputs = []
        labels = []

        for x in data_x:
            inputs.append([x, x])
            labels.append(0)

            if x == 1 - x:
                continue

            inputs.append([x, 1 - x])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape((-1, 1))

    @staticmethod
    def fetch_data(mode, n):
        """ Data gather interface

        Args:
            mode (str): 'Linear' or 'XOR', indicate which generator is used.
            n (int):    the number of data points generated in total.
        """
        assert mode == 'Linear' or mode == 'XOR'

        data_gen_func = {
            'Linear': GenData._gen_linear,
            'XOR': GenData._gen_xor
        }[mode]

        return data_gen_func(n)


class SimpleNet:
    def __init__(self, hidden_size, num_step=2000, print_interval=100):
        """ A hand-crafted implementation of simple network.

        Args:
            hidden_size:    the number of hidden neurons used in this model.
            num_step (optional):    the total number of training steps.
            print_interval (optional):  the number of steps between each reported number.
        """
        self.num_step = num_step
        self.print_interval = print_interval

        # Model parameters initialization
        # Please initiate your network parameters here.
        
        self.inputSize = 2
        self.outputSize = 1
        """
        self.L1 = 4
        self.L2 = 4
        """
        self.W1 = np.random.randn(self.inputSize, hidden_size)
        self.W2 = np.random.randn(hidden_size, hidden_size)
        self.W3 = np.random.randn(hidden_size, self.outputSize)

    @staticmethod
    def plot_result(data, gt_y, pred_y):
        """ Data visualization with ground truth and predicted data comparison. There are two plots
        for them and each of them use different colors to differentiate the data with different labels.

        Args:
            data:   the input data
            gt_y:   ground truth to the data
            pred_y: predicted results to the data
        """
        assert data.shape[0] == gt_y.shape[0]
        assert data.shape[0] == pred_y.shape[0]

        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title('Ground Truth', fontsize=18)

        for idx in range(data.shape[0]):
            if gt_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.subplot(1, 2, 2)
        plt.title('Prediction', fontsize=18)

        for idx in range(data.shape[0]):
            if pred_y[idx] == 0:
                plt.plot(data[idx][0], data[idx][1], 'ro')
            else:
                plt.plot(data[idx][0], data[idx][1], 'bo')

        plt.show()


    def forward(self, inputs):
        """ Implementation of the forward pass.
        It should accepts the inputs and passing them through the network and return results.
        """
        self.inputs = inputs
        # Layer 1  1*2, 2*100
        self.L1_inp = np.dot(inputs, self.W1)
        self.L1_out = sigmoid(self.L1_inp)

        # Layer 2 1*100, 100*100
        self.L2_inp = np.dot(self.L1_out, self.W2)
        self.L2_out = sigmoid(self.L2_inp)
        
        # output 1*100, 100*1
        self.output_inp = np.dot(self.L2_out, self.W3)
        outputs = sigmoid(self.output_inp)

        return outputs

    def backward(self, y, y_hat):
        """ Implementation of the backward pass.
        It should utilize the saved loss to compute gradients and update the network all the way to the front.
        """

        if(y_hat > y):
            d = 1
        else:
            d = -1
        lr = 0.1

        self.outputs_delta = d*der_sigmoid(self.output_inp)
        L3_layer = np.dot(self.L2_out.T, self.outputs_delta)

        self.L2_error = np.dot( self.W3, self.outputs_delta)
        der_tmp = der_sigmoid(self.L2_inp)
        self.L2_delta = der_tmp * self.L2_error.T
        L2_layer = np.dot(self.L1_out.T, self.L2_delta) 
        
        self.L1_error = np.dot( self.L2_delta, self.W2.T)
        L1_der_tmp = der_sigmoid(self.L1_inp)
        self.L1_delta = L1_der_tmp * self.L1_error
        L1_layer = np.dot(self.inputs.T, self.L1_delta) 

        self.W3 = self.W3 - lr*L3_layer
        self.W2 = self.W2 - lr*L2_layer
        self.W1 = self.W1 - lr*L1_layer


    def train(self, inputs, labels):
        """ The training routine that runs and update the model.

        Args:
            inputs: the training (and testing) data used in the model.
            labels: the ground truth of correspond to input data.
        """
        # make sure that the amount of data and label is match
        assert inputs.shape[0] == labels.shape[0]

        n = inputs.shape[0]
        for epochs in range(self.num_step):
            for idx in range(n):
                # operation in each training step:
                #   1. forward passing
                #   2. compute loss
                #   3. propagate gradient backward to the front
                self.output = self.forward(inputs[idx:idx+1, :])
                self.error = self.output - labels[idx:idx+1, :]
                self.backward(labels[idx:idx+1, :], self.output)

            if epochs % self.print_interval == 0:
                print('Epochs {}: '.format(epochs))
                self.test(inputs, labels)

        print('Training finished')
        self.test(inputs, labels)

    def test(self, inputs, labels):
        """ The testing routine that run forward pass and report the accuracy.

        Args:
            inputs: the testing data. One or several data samples are both okay.
                The shape is expected to be [BatchSize, 2].
            labels: the ground truth correspond to the inputs.
        """
        n = inputs.shape[0]
        error = 0.0
        for idx in range(n):
            result = self.forward(inputs[idx:idx+1, :])
            error += abs(result - labels[idx:idx+1, :])

        error /= n
        print('accuracy: %.2f' % ((1 - error)*100) + '%')
        print('')


if __name__ == '__main__':
    data, label = GenData.fetch_data('XOR', 70)

    net = SimpleNet( hidden_size=100)
    # net = SimpleNet(num_step=100, hidden_size=100)
    net.train(data, label)

    pred_result = np.round(net.forward(data))

    SimpleNet.plot_result(data, label, pred_result)

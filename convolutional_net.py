
class ConvolutionalNet:
    def __init__(self):
        pass

    def convolution(self, x, kernel, stride=1):
        kernel = np.array(kernel)

        conv_x = list()
        temp_result = list()

        # padding
        n_columns = image.shape[1]
        vzeros = np.zeros(n_columns)
        image = np.vstack((image, vzeros))
        image = np.vstack((vzeros, image))

        n_rows = image.shape[0]
        hzeros = np.zeros((n_rows, 1))
        image = np.hstack((image, hzeros))
        image = np.hstack((hzeros, image))

        n_rows = image.shape[0]
        n_columns = image.shape[1]

        for x in range(1, n_rows - 1, stride):
            for y in range(1, n_columns - 1, stride):
                row_start = x - 1
                row_finish = row_start + (kernel.shape[0])
                column_start = y - 1
                column_finish = column_start + (kernel.shape[1])

                region = x[row_start:row_finish, column_start:column_finish]

                result = np.multiply(region, kernel)
                result = np.sum(result)

                temp_result.append(result)

            conv_x.append(temp_result.copy())
            temp_result[:] = []

        return np.array(conv_x)

    def forward_step(self):
        pass

    def max_pooling(self):
        pass

class Model:
    def __init__(self):
        self.dataset = [[1, 1], [3, -1]]
        self.x_t = 0
        self.context_units = [0, 0, 0]
        self.hidden_layer = [0, 0, 0]
        self.output = 0

        self.params = [0.5] * 15

    def MSE(self):
        sum = 0
        for i in self.dataset:
            sum += (i[1] - i[0]) ** 2
        return sum / len(self.dataset)


    def forward(self, in_):
        t1 = self.params[0] * in_ + self.params[1] * self.context_units[0] + self.params[2] * self.context_units[1] + self.params[3] * self.context_units[2]
        t2 = self.params[4] * in_ + self.params[5] * self.context_units[0] + self.params[6] * self.context_units[1] + self.params[7] * self.context_units[2]
        t3 = self.params[8] * in_ + self.params[9] * self.context_units[0] + self.params[10] * self.context_units[1] + self.params[11] * self.context_units[2]
        self.context_units[0] += self.activation_function(t1)
        self.context_units[1] += self.activation_function(t2)
        self.context_units[2] += self.activation_function(t3)

        out_ = self.activation_function(t1) * self.params[12] + self.activation_function(t2) * self.params[13] + self.activation_function(t3) * self.params[14]
        out_ = self.activation_function(out_)
        return out_

    def activation_function(self, inp):
        # relu
        if (inp >= 0):
            return inp
        else:
            return 0

    def train(self, data):
        pass



def main():
    model = Model()
    print(model.forward(5))


if __name__ == "__main__":
    main()
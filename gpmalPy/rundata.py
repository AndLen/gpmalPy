class RD(object):
    def __init__(self):
        self.max_depth = 8  # 7#12#8
        self.max_height = 14  # 10#17#14
        self.pop_size = 1024#00  # 1024#100
        self.cxpb = 0.8
        self.mutpb = 0.2
        self.elitism = 10
        self.warnOnce = False
        self.use_ercs = True
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


rd = RD()


# fitnessCache = None

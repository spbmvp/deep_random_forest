from ForestsModel import ForestsModel

if __name__ == '__main__':
    print("hello")
    cascade_forest = ForestsModel().get_forests()
    cascade_forest.fit()

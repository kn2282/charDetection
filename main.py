# This is a sample Python script.
from NeuralNetwork import CharDetector
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    charDetector = CharDetector((20, 20), 4)

    train_pairs = []
    for index_l, l in enumerate(['a', 'k', 'n', 't']):
        for i in range(8):
            train_pairs.append( ("asset/" + l + str(i+1) + ".jpg", index_l) )
    charDetector.load_train_images(train_pairs)

    accuracy_pairs = []
    for index_l, l in enumerate(['a', 'k', 'n', 't']):
        for i in range(8, 9):
            accuracy_pairs.append( ("asset/" + l + str(i+1) + ".jpg", index_l) )
    charDetector.load_accuracy_images(accuracy_pairs)

    charDetector.train()
    print(charDetector.check_accuracy())

    print(charDetector.predict("asset/t10.jpg"))






import pickle

file = open("terms.pickle",'rb')
object_file = pickle.load(file)
print(object_file)
file.close()
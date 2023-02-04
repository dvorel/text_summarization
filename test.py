from csvDataset import csvDataset

#test dataset with train.csv
if __name__=="__main__":
  DATA = "datasets/cnn_dailymail/train.csv"
  dataset = csvDataset(DATA)
  print(dataset[0]["text"])
  print("\nSUMMARY: \n")
  print(dataset[0]["summary"])

#TODO: model test

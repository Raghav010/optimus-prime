## **Usage**
To run the transformer script : `python transformer.py epochs batchSize head_count dropout embedSize`

**Recommended Settings**
- epochs=10
- batchSize=10
- head_count=4
- dropout=0.1
- embedSize=300

---

**Make sure a directory called `transformerModels` is present**


- The script will output Bleu Scores files for train and test datasets
- Each line of the Bleu scores file corresponds to the bleu score of that sentence
- The last line of the Bleu scores file is the average bleu score across all sentences in the dataset
- It outputs a csv file containing the train and validation losses for each epoch
- It also outputs a file containing the test loss


## **Loading the Model**
`model.load_state_dict(torch.load("PATH"))`


#### **Model Link**
https://iiitaphyd-my.sharepoint.com/:u:/g/personal/raghav_donakanti_students_iiit_ac_in/Ec-O0Jd-2-NCj9mxkbIgo8gBVLUW06RZb6_It9SBC3Ub8w?e=PKaQmC


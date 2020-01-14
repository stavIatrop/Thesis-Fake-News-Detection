import matplotlib.pyplot as plt

f = open("KNN_max_df", "r")
y1 = list()
y2 = list()
x = list()
for i in range(20, 91, 5):

    x.append(i * 0.01)
    first = f.readline()
    scnd = f.readline()
    third = f.readline()


    acc = scnd
    acc.replace("\n", "\0")
    acc = float(acc.strip(' "'))
    y1.append(acc)

   
    f1 = third
    f1.replace("\n", "\0")
    f1 = float(f1.strip(' "'))
    y2.append(f1)


f.close() 

# plt.xlabel('Max_df')
# plt.ylabel('Score')
# plt.title('RF: Scores in relation with max_df ')
# plt.axis([min(x)-0.05, max(x) + 0.05, min(min(y1), min(y2))-0.0001, max(max(y1), max(y2))+0.0001])
# plt.plot( x, y1, 'rx', label = 'Accuracy' )
# plt.plot( x, y2, 'gx', label = 'F1 score' )


# plt.grid(True)
# plt.legend(loc='upper right')
# plt.savefig('RF_graph_gossipcop.png',bbox_inches='tight')


# plt.clf()

plt.xlabel('Max_df')
plt.ylabel('Score')
plt.title('DT: Scores in relation with max_df ')
plt.axis([min(x)-0.05, max(x) + 0.05, min(min(y1), min(y2))-0.0001, max(max(y1), max(y2))+0.0001])
plt.plot( x, y1, 'rx', label = 'Accuracy' )
plt.plot( x, y2, 'gx', label = 'F1 score' )


plt.grid(True)
plt.legend(loc='upper right')
plt.savefig('DT_graph_gossipcop.png',bbox_inches='tight')


# plt.xlabel('Max_df')
# plt.ylabel('Precision')
# plt.title('SVM: Precision score in relation with max_df ')
# plt.axis([min(x)-0.05, max(x) + 0.05, min(y1)-0.0001, max(y1)+0.0001])
# plt.plot( x, y1, 'rx' )
# plt.grid(True)
# plt.savefig('Precision_svm_graph_politifact.png',bbox_inches='tight')

# plt.clf()

# plt.xlabel('Max_df')
# plt.ylabel('Recall')
# plt.title('SVM: Recall score in relation with max_df')
# plt.axis([min(x)-0.05, max(x) + 0.05, min(y2)-0.0001, max(y2)+0.0001])
# plt.plot(x, y2, 'rx')
# plt.grid(True)
# plt.savefig('Recall_svm_graph_politifact.png',bbox_inches='tight')

# plt.clf()

# plt.xlabel('Max_df')
# plt.ylabel('Accuracy')
# plt.title('SVM: Accuracy score in relation with max_df')
# plt.axis([min(x)-0.05, max(x) + 0.05, min(y3)-0.0001, max(y3)+0.0001])
# plt.plot(x, y3, 'rx')
# plt.grid(True)
# plt.savefig('Accuracy_svm_graph_politifact.png',bbox_inches='tight')

# plt.clf()

# plt.xlabel('Max_df')
# plt.ylabel('F1 score')
# plt.title('SVM: F1_score in relation with max_df')
# plt.axis([min(x)-0.05, max(x) + 0.05, min(y4)-0.0001, max(y4)+0.0001])
# plt.plot( x, y4, 'rx')
# plt.grid(True)
# plt.savefig('F1_score_svm_graph_politifact.png',bbox_inches='tight')

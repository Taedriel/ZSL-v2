# THIS FILE IS NOT INTENDED FOR USE IN THE PIPELINE AND IS ONLY FOR SEPARATE TESTING

from fsl_classification import constants, classes_for_model

CUB, IMAGES, OMNIGLOT = False, True, False
PATH_DATA = ""
listClass = []

if CUB or IMAGES:
  PATH_DATA = HEAD+"pipeline/model/data/CUB/images/" if CUB else HEAD+"pipeline/images/"
  listClass = os.listdir(PATH_DATA)
elif OMNIGLOT:
  PATH_DATA = HEAD+"pipeline/model/dataO/omniglot-py/images_background/" 
  listAlphabet = os.listdir(PATH_DATA)
  choosenAlphabet = listAlphabet[random.randint(0, len(listAlphabet)-1)]
  listClass = [choosenAlphabet+"/"+char for char in os.listdir(PATH_DATA+choosenAlphabet)]


supportClasses = getNrandomClassesPaths(PATH_DATA, listClass, N_WAY)

print(supportClasses)
supportSet, querySet = getSets(supportClasses, N_SHOT, N_QUERY)

justSupport = getOnlyImages(supportSet)
justQuery = getOnlyImages(querySet)
plot_images(justSupport, title="support set", images_per_row=N_SHOT)
plot_images(justQuery, title="query set", images_per_row=N_QUERY)

training_model = Trainer(PATH_MODEL, model, False)

raining_model.resetModel(reset_by_param=False)

trainingNeeded = True
if trainingNeeded:
  losses = training_model.training(supportSet, (0, 0), 0)

showData(losses, "loss during training", 3)

evaluation_model = Tester(training_model.model)
accuracyResults, y_pred, y, correctPreds, incorrectPreds, indexIncorrectQuery = evaluation_model.evaluateWithMetric(supportSet, querySet)

_, _, confM = getMatrixReport(y, y_pred)
print(accuracyResults)
print(confM)
print(indexIncorrectQuery)
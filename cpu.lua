-- deps
require 'nn'

-- load data
file = torch.DiskFile('dat/facies_vectors.t7', 'r')
facies = file:readObject()
file:close()
file = torch.DiskFile('dat/validation_data_nofacies.t7', 'r')
validate = file:readObject()
file:close()

-- build dicts
--print("facies size: ", facies:size()[1], "x", facies:size()[2])
--print("validate size: ", validate:size()[1], "x", validate:size()[2])

	-- initialize
training_data = {}
testing_data = {}
depth = {}

	-- build the training wells into the table
training_data["shrimplin"] = facies[{{1,471},{3,9}}]
training_data["alexander"] = facies[{{472,937},{3,9}}]
training_data["shankle"] = facies[{{938,1386},{3,9}}]
training_data["luke"] = facies[{{1387,1847},{3,9}}]
training_data["kimzey"] = facies[{{1848,2286},{3,9}}]
training_data["cross"] = facies[{{2287,2787},{3,9}}]
training_data["nolan"] = facies[{{2788,3202},{3,9}}]
training_data["recruit"] = facies[{{3203,3282},{3,9}}]
training_data["newby"] = facies[{{3283,3745},{3,9}}]
training_data["churchman"] = facies[{{3746,4149},{3,9}}]

	-- build the testing wells into the table
testing_data["stuart"] = validate[{{1,474},{2,8}}]
testing_data["crawford"] = validate[{{475,830},{2,8}}]

	-- build a depth log for plotting
depth["shrimplin"] = facies[{{1,471},{2}}]
depth["alexander"] = facies[{{472,937},{2}}]
depth["shankle"] = facies[{{938,1386},{2}}]
depth["luke"] = facies[{{1387,1847},{2}}]
depth["kimzey"] = facies[{{1848,2286},{2}}]
depth["cross"] = facies[{{2287,2787},{2}}]
depth["nolan"] = facies[{{2788,3202},{2}}]
depth["recruit"] = facies[{{3203,3282},{2}}]
depth["newby"] = facies[{{3283,3745},{2}}]
depth["churchman"] = facies[{{3746,4149},{2}}]
depth["stuart"] = validate[{{1,474},{1}}]
depth["crawford"] = validate[{{475,830},{1}}]

-- normalize the data
	-- training data
mean = {}
stdv = {}

for key,value in pairs(training_data) do --over each well
    mean[key] = torch.Tensor(7)
    stdv[key] = torch.Tensor(7)
    for i = 1, 7 do --over each log
        mean[key][i] = training_data[key][{{},{i}}]:mean()
        training_data[key][{{},{i}}]:add(-mean[key][i])
        
        stdv[key][i] = training_data[key][{{},{i}}]:std()
        training_data[key][{{},{i}}]:div(stdv[key][i])
    end
end
	-- validation data
mean = {}
stdv = {}

for key,value in pairs(testing_data) do --over each well
    mean[key] = torch.Tensor(7)
    stdv[key] = torch.Tensor(7)
    for i = 1, 7 do --over each log
        mean[key][i] = testing_data[key][{{},{i}}]:mean()
        testing_data[key][{{},{i}}]:add(-mean[key][i])
        
        stdv[key][i] = testing_data[key][{{},{i}}]:std()
        testing_data[key][{{},{i}}]:div(stdv[key][i])
    end
end

-- facies labels for training
facies_labels = {}

facies_labels["shrimplin"] = facies[{{1,471},{1}}]
facies_labels["alexander"] = facies[{{472,937},{1}}]
facies_labels["shankle"] = facies[{{938,1386},{1}}]
facies_labels["luke"] = facies[{{1387,1847},{1}}]
facies_labels["kimzey"] = facies[{{1848,2286},{1}}]
facies_labels["cross"] = facies[{{2287,2787},{1}}]
facies_labels["nolan"] = facies[{{2788,3202},{1}}]
facies_labels["recruit"] = facies[{{3203,3282},{1}}]
facies_labels["newby"] = facies[{{3283,3745},{1}}]
facies_labels["churchman"] = facies[{{3746,4149},{1}}]

-- chop out blind well
blind_well = {}
blind_labels = {}

blind_well["newby"] = training_data["newby"][{{},{}}]

training_data["newby"] = nil

blind_labels["newby"] = facies_labels["newby"][{{},{}}]

facies_labels["newby"] = nil

-- build the neural net ----------------------------------------
net = nn.Sequential()
net:add(nn.Linear(7,20))
net:add(nn.Tanh())
net:add(nn.Linear(20,9))
net:add(nn.Tanh())
--net:add(nn.LogSoftMax())
----------------------------------------------------------------

-- test the net -> forward
temp = torch.Tensor(7)
for i = 1,7 do
    temp[i] = training_data["shrimplin"][1][i]
end
input = temp

output = net:forward(input)

--print("forward output =\n", output)
--print("correct facies = ", facies_labels["shrimplin"][1])

-- calibrate gradient parameters
net:zeroGradParameters()

gradInput = net:backward(input, torch.rand(9))

-- define the loss function
criterion = nn.CrossEntropyCriterion()

criterion:forward(output,facies_labels["shrimplin"][1])
--print(criterion:forward(output,facies_labels["shrimplin"][1]))

gradients = criterion:backward(output, facies_labels["shrimplin"][1])
--print("gradients = ", gradients)

gradInput = net:backward(input, gradients)
print(gradInput)

-- condition the data
trainset = {}

	-- the data
trainset["data"] = torch.Tensor(facies:size()[1]-blind_well["newby"]:size()[1],7) 

idx = 0
for key,value in pairs(training_data) do
    for i = 1,training_data[key]:size()[1] do
        trainset["data"][i + idx] = training_data[key][i]
    end
    idx = idx + training_data[key]:size()[1]
end

	-- the answers
trainset["facies"] = torch.Tensor(facies:size()[1]-blind_labels["newby"]:size()[1])

idx = 0
for key,value in pairs(facies_labels) do
    for i = 1, facies_labels[key]:size()[1] do
        trainset["facies"][i + idx] = facies_labels[key][i]
    end
    idx = idx + facies_labels[key]:size()[1]
end


-- write index() and size() functions
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.facies[i]} 
                end}
);

function trainset:size() 
    return self.data:size(1) 
end

-- train the net
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.00001
trainer.maxIteration = 10

print("starting training")
timer = torch.Timer()
trainer:train(trainset)
print("training time =", timer:time().real)

-- predict using the net
	-- condition the testing data
testset = {}

	-- the data
testset["data"] = torch.Tensor(blind_well["newby"]:size()[1],7) 

for i = 1,blind_well["newby"]:size()[1] do
    testset["data"][i] = blind_well["newby"][i]
end

	-- the answers
testset["facies"] = torch.Tensor(blind_labels["newby"]:size()[1])

for i = 1, blind_labels["newby"]:size()[1] do
    testset["facies"][i] = blind_labels["newby"][i]
end

setmetatable(testset, 
    {__index = function(t, i) 
                    return {t.data[i], t.facies[i]} 
                end}
);

function testset:size() 
    return self.data:size(1) 
end

-- calculate the accuracy
correct = 0
for i=1,testset:size() do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end
print("correct: ", correct, 100*correct/testset:size() .. ' % ')

class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,testset:size() do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end
classes = {'SS','CSiS','FSiS','SiSh','MS','WS','D','PS','BS'}
for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end


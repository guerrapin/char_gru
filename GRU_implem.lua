require 'nngraph'
require 'gnuplot'

local model_utils = require 'model_utils'
local utils = require 'utils'
local CharLMMinibatchLoader=require 'CharLMMinibatchLoader'

---------------------------------------------------------
------------- COMMAND OPTIONS ---------------------------
---------------------------------------------------------
cmd = torch.CmdLine()
cmd:option('-batch_size',100,"mini-batch size")
cmd:option('-maxEpoch',10,"number of epochs")
cmd:option('-learning_rate',1e-2,"learning rate")
cmd:option('-seq_length',5,"number of characters per sequence")
cmd:option('-ratio',0.8,"train/total ratio. To split the dataset in train and test sets")
cmd:option('-latent_dim',100,"dimension of the latent state of the RNN")
cmd:option('-data','data.t7','tensor containing the pre-formated data')
cmd:option('-vocab','vocab.t7','translation between characters and indexes')

local opt = cmd:parse(arg)
print("Parameters of this experiment :")
print(opt)

---------------------------------------------------------
-------------- DATA RESHAPING STEP ----------------------
---------------------------------------------------------

local data=CharLMMinibatchLoader.create(opt.data,opt.vocab,opt.batch_size,opt.seq_length)

local vocab_size = data.vocab_size
local n_batches = table.getn(data.x_batches)

-- Split the data in train and test sets
local train_size = torch.floor(n_batches*opt.ratio)
local xs_train = utils.subrange(data.x_batches,1,train_size)
local ys_train = utils.subrange(data.y_batches,1,train_size)
local xs_test = utils.subrange(data.x_batches, train_size + 1, n_batches)
local ys_test = utils.subrange(data.y_batches, train_size + 1, n_batches)

function prepareBatch(xin_batch, yin_batch)
   --[[xin_batch and yin_batch format : batch_size x seq_length
       xout_batch format : seq_length (table) x batch_size x vocab_size
       yout_batch format : seq_length (table) x batch_size
       a table is required because of the input format of the network
       transform x_batch into a one-hot vecteur whose dimension is the vocabulary length
   ]]--
   local xout_batch = {}
   local yout_batch = {}

   for j=1, opt.seq_length do
      table.insert(xout_batch,torch.zeros(opt.batch_size,vocab_size))
      table.insert(yout_batch,torch.Tensor(opt.batch_size))
   end

   for index = 1,opt.batch_size do
      for j=1, opt.seq_length do
         xout_batch[j][index][xin_batch[index][j]] = 1
         yout_batch[j][index] = yin_batch[index][j]
      end
   end
   return xout_batch, yout_batch
end

---------------------------------------------------------
-------------- RNN GENERATION STEP (GRU) ----------------
---------------------------------------------------------

-- Create one GRU Unit
x = nn.Identity()()
ht_1 = nn.Identity()()
Wr = nn.Linear(vocab_size,opt.latent_dim)(x); Wz = nn.Linear(vocab_size,opt.latent_dim)(x)
Ur = nn.Linear(opt.latent_dim,opt.latent_dim)(ht_1); Uz = nn.Linear(opt.latent_dim,opt.latent_dim)(ht_1)
sr = nn.CAddTable()({Ur,Wr}); sz = nn.CAddTable()({Uz,Wz})
rt = nn.Sigmoid()(sr); zt = nn.Sigmoid()(sz)
rt2 = nn.CMulTable()({rt, ht_1}); rt3 = nn.Linear(opt.latent_dim,opt.latent_dim)(rt2)
htilde = nn.Tanh()(nn.CAddTable()({rt3,nn.Linear(vocab_size,opt.latent_dim)(x)}))
z_1 = nn.AddConstant(1)(nn.MulConstant(-1)(zt))
ht = nn.CAddTable()({nn.CMulTable()({htilde,zt}),nn.CMulTable()({ht_1,z_1})})
GRU = nn.gModule({x,ht_1},{ht})

-- Generate as many clones as the number of elements in a sequence
local modelGRU = model_utils.clone_many_times(GRU,opt.seq_length)

-- Generate also clones of a decode graph
-- decod node convert an output state into a log-probability distribution over the vocabulary
z = nn.Identity()()
output = nn.LogSoftMax()(nn.Linear(opt.latent_dim,vocab_size)(z))
Decod = nn.gModule({z},{output})
local modelDecod = model_utils.clone_many_times(Decod,opt.seq_length)

-- Plug the clones together in a super graph
local xi = {} -- input nodes for sequence of char
local z = nn.Identity()(); xi[1] = z
local Di = {} -- GRU decod nodes (output)
for i=1, opt.seq_length do
   xi[i+1] = nn.Identity()()
   z = modelGRU[i]({xi[i+1],z})
   Di[i] = modelDecod[i](z)
end
local full_model = nn.gModule(xi,Di)

--graph.dot(full_model.fg, 'full','FULLgraphfilename')

-- Create parallel criterions
local crit = nn.ParallelCriterion()
for i=1, opt.seq_length do
   crit:add(nn.ClassNLLCriterion())
end

---------------------------------------------------------
-------------- LEARNING AND EVALUATION ------------------
---------------------------------------------------------

function Eval(xtest,ytest)
   -- evaluate accuracy on inference given a test set

   local test_size = table.getn(xtest)

   local acc = {}; for i=1,opt.seq_length do table.insert(acc,0) end

   -- iterate over the batches
   for j = 1, test_size do

      xbatch, targ = prepareBatch(xtest[j], ytest[j])
      -- generate initialisation of the latent state and concatenate to the input
      local init_latent_space = torch.zeros(opt.batch_size,opt.latent_dim)
      local input = utils.TableConcat({init_latent_space},xbatch)

      -- compute the output
      output = full_model:forward(input)

      for num_seq =1, opt.seq_length do
         local distrib = torch.exp(output[num_seq]) -- exponential because LogSoftMax was used
         local index_picked = torch.multinomial(distrib,1)

         acc[num_seq] = acc[num_seq] + utils.accuracy(index_picked:double(),targ[num_seq])/test_size
      end
   end

   return acc
end


-- formatting stuff for verbose on terminal
text = {}
prev = ''
for i=1,opt.seq_length do
   prev = 'w' .. i .. ',' .. prev
   table.insert(text,'P(w' .. i+1 .. '|' .. prev .. ')')
end
print("\nMean accuracy on test set for the following probabilities")
print(unpack(text))


local train_losses = {}
local test_accuracy = {}

-- loop over the epochs
for iteration=1,opt.maxEpoch do

   -- Test Phase

   local mean_acc = Eval(xs_test, ys_test)
   print(unpack(mean_acc))
   table.insert(test_accuracy, mean_acc)

   -- Train Phase

   local shuffle = torch.randperm(train_size)

   -- loop over the batches
   for j = 1, train_size do
      x_batch, target = prepareBatch(xs_train[shuffle[j]], ys_train[shuffle[j]])

      local init_latent_space = torch.zeros(opt.batch_size,opt.latent_dim)
      local input = utils.TableConcat({init_latent_space},x_batch)

      full_model:zeroGradParameters()

      -- forward pass
      output = full_model:forward(input)
      loss = crit:forward(output,target)
      table.insert(train_losses,loss)

      -- backward pass
      delta = crit:backward(output,target)

      full_model:backward(input,delta)

      -- update parameters
      full_model:updateParameters(opt.learning_rate)
   end
   gnuplot.plot('Train loss', torch.Tensor(train_losses))
end

table.insert(test_accuracy,Eval(xs_test,ys_test))


--gnuplot.plot(torch.Tensor(train_losses))
--gnuplot.plot(torch.Tensor(test_accuracy))

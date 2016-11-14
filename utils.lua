require 'torch'

local utils = {}

function utils.subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end


function utils.TableConcat(t1,t2)
   for i=1,#t2 do
      t1[#t1+1] = t2[i]
   end
   return t1
end

function utils.accuracy(y, out)
   -- return the accuracy of the prediction "out" compared to the label "y" (i.e. a value between 0 and 1)

   accuracy_value = torch.mean(torch.eq(torch.sign(out),y):double())

   return accuracy_value

end

return utils

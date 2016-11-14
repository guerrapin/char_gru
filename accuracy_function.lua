

function accuracy(y, out)
   -- return the accuracy of the prediction "out" compared to the label "y" (i.e. a value between 0 and 1)

   accuracy_value = torch.mean(torch.eq(torch.sign(out),y):double())

   return accuracy_value

end

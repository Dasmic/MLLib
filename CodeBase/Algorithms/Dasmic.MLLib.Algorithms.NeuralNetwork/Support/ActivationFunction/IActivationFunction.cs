using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction
{
    public interface IActivationFunction
    {
        double GetValue(double weightValueProduct);
        double GetDerivativeValue(double weightValueProduct);
    }
}

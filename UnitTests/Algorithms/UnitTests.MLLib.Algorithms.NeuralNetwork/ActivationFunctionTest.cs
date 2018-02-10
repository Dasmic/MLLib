using Dasmic.MLLib.Algorithms.NeuralNetwork;
using Dasmic.Portable.Core;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;

namespace UnitTests.MLLib.Algorithms.NeuralNetwork
{
    [TestClass]
    public class ActivationFunctionTest : BaseTest
    {
        [TestMethod]
        public void NN_activation_function_linear()
        {
            Linear aFunc = new Linear();
            double value = aFunc.GetValue(3.0);
            Assert.IsTrue(SupportFunctions.DoubleCompare(value, 3.00));
        }


        [TestMethod]
        public void NN_activation_function_hyperbolictangent()
        {
            HyperbolicTangent aFunc = new HyperbolicTangent();
            double value = aFunc.GetValue(3.0);
            Assert.IsTrue(SupportFunctions.DoubleCompare(value, 0.9950));
        }

        [TestMethod]
        public void NN_activation_function_sigmoid()
        {
            Sigmoid aFunc = new Sigmoid();
            double value = aFunc.GetValue(3.0);
            Assert.IsTrue(SupportFunctions.DoubleCompare(value, 0.9526));
        }

        [TestMethod]
        public void NN_activation_function_step_1()
        {
            Step aFunc = new Step();
            double value = aFunc.GetValue(3.0);
            Assert.AreEqual(value, 1);
        }

        [TestMethod]
        public void NN_activation_function_step_0()
        {
            Step aFunc = new Step();
            double value = aFunc.GetValue(0-3.0);
            Assert.AreEqual(value, -1);
        }
    }
}

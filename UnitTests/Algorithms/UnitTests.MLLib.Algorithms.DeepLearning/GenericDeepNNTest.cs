using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.DeepLearning;
using Dasmic.MLLib.Algorithms.NeuralNetwork;
using Dasmic.MLLib.Common.MLCore;

namespace UnitTests.MLLib.Algorithms.DeepLearning
{
    [TestClass]
    public class GenericDeepNNTest:BaseTest
    {
        [TestMethod]
        public void Deep_NN_generic_jason_simple_rmse()
        {
            Init_dataset_jason_linear_regression();
            BuildGenericDeepNN build =
                    new BuildGenericDeepNN();

            build.SetParameters(0, 2  );
            //Use Default Parameters
            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);
            
            double value = model.GetModelRMSE(_trainingData);

            Assert.IsTrue(value < .61 && value > 0);
        }
    }
}

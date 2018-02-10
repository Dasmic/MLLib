using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.NeuralNetwork;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.Portable.Core;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.NeuralNetwork
{
    [TestClass]
    public class GenericBackPropagationStandardTest : BaseTest
    {
    #region Training Data from Jason Gaussian Naive Bayes

        [TestMethod]
        public void NN_backpropagation_generic_std_gnb_single_training_sample_class_0()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();

            build.SetParameters(1,1);
            
            build.AddHiddenLayer(0, 2, new Sigmoid());
            build.SetOutputLayerActivationFunction(new Sigmoid());

            ModelBackPropagationBase model =
                (ModelBackPropagationBase )build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int row = 0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }

        [TestMethod]
        public void NN_backpropagation_generic_std_gnb_all_training_samples()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();
            build.SetParameters(1,1);                        
            build.AddHiddenLayer(0, 2, new Sigmoid());
            build.SetOutputLayerActivationFunction(new Sigmoid());

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count = 0;
            for (int row = 0; row < 10; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                if (SupportFunctions.DoubleCompare(value,
                        _trainingData[_indexTargetAttribute][row]))
                    count++;
            }

            //Due to random weights
            Assert.IsTrue(count >= 8 && count <= 10);
        }

        [TestMethod]
        public void NN_backpropagation_generic_std_3L_gnb_all_training_samples()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();
            build.SetParameters(1,2,.5,1500);
            //build.SetNumberOfHiddenLayers(2);
            build.AddHiddenLayer(0, 2, new Sigmoid());
            build.AddHiddenLayer(1, 2, new Sigmoid());
            build.SetOutputLayerActivationFunction(new Sigmoid());

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count = 0;
            for (int row = 0; row < 10; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                if (SupportFunctions.DoubleCompare(value,
                        _trainingData[_indexTargetAttribute][row]))
                    count++;
            }
            //Due to random weights
            Assert.IsTrue(count >= 5);
        }


        [TestMethod]
        [ExpectedException(typeof(InvalidNeuralNetworkLayer))]
        public void NN_backpropagation_generic_std_invalid_layer_throws_exception()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();
            build.SetParameters(1,1, .5, 1500);            
            build.AddHiddenLayer(0, 2, new Sigmoid());
            build.AddHiddenLayer(1, 2, new Sigmoid());
            build.SetOutputLayerActivationFunction(new Sigmoid());
            
        }

        [TestMethod]
        [ExpectedException(typeof(NeuralNetworkConfigurationNotReady))]
        public void NN_backpropagation_generic_std_missing_layer_throws_exception()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();
            build.SetParameters(1,3, .5, 1500);
            //build.SetNumberOfHiddenLayers(3);
            build.AddHiddenLayer(0, 2, new Sigmoid());
            build.AddHiddenLayer(1, 2, new Sigmoid());
            build.SetOutputLayerActivationFunction(new Sigmoid());
         
            ModelBackPropagationBase model =
                  (ModelBackPropagationBase)build.BuildModel(
                      _trainingData, _attributeHeaders,
                      _indexTargetAttribute);           
        }

        #endregion

        #region Training Data from Jason Naive Bayes

        [TestMethod]
        public void NN_backpropagation_generic_std_nb_all_training_samples()
        {
            initData_dataset_naive_bayes_jason_example();
            BuildGenericBackPropagationStandard build =
                     new BuildGenericBackPropagationStandard();
            build.SetParameters(1,1);
            //build.SetNumberOfHiddenLayers(1);
            build.AddHiddenLayer(0, 2, new Sigmoid());
            build.SetOutputLayerActivationFunction(new Sigmoid());


            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int count = 0;
            for (int row = 0; row < 10; row++)
            {
                double[] data = GetSingleTrainingRowDataForTest(row);
                double value = model.RunModelForSingleData(data);

                if (SupportFunctions.DoubleCompare(value,
                        _trainingData[_indexTargetAttribute][row]))
                    count++;
            }

            //Due to random weights
             Assert.IsTrue(count >= 8 && count <= 10);
        }
        #endregion

        #region Regression (Mode=0) Tests
     
        [TestMethod]
        public void NN_backpropagation_generic_std_jason_simple_rmse()
        {
            Init_dataset_jason_linear_regression();
            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();
            build.SetParameters(0,1,.05,2000,.05);
            //build.SetNumberOfHiddenLayers(1);
            build.AddHiddenLayer(0, 2, new Sigmoid());            
            build.SetOutputLayerActivationFunction(new Linear());
           
            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            double value = model.GetModelRMSE(_trainingData);

            Assert.IsTrue(value < 1.0);            
        }

        /// <summary>
        /// R:
        /// tdata=read.csv("C:\\Dasmic\\Development\\NET\\MLLib.NET\\DataSets\\pythagoras.csv",header=T)
        /// library(neuralnet)
        /// nn <- neuralnet(H ~ P+B,tdata, hidden=c(2), linear.output=T)
        /// </summary>
        [TestMethod]
        public void NN_backpropagation_generic_std_one_hidden_pythagoras_rmse()
        {
            Init_dataset_pythagoras();
            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();
            build.SetParameters(0,1,.3, 4000);
            //build.SetNumberOfHiddenLayers(1);
            build.AddHiddenLayer(0, 2, new Sigmoid());
            build.SetOutputLayerActivationFunction(new Linear());

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            double value = model.GetModelRMSE(_trainingData);

            Assert.IsTrue(value < 10.00);
        }


        /// <summary>
        /// R:
        /// tdata=read.csv("C:\\Dasmic\\Development\\NET\\MLLib.NET\\DataSets\\pythagoras.csv",header=T)
        /// library(neuralnet)
        /// weights<-c(.005,.005,.005,.005,.005,.005,.005,.005,.005)
        /// 
        /// nn <- neuralnet(H ~ P+B,tdata, algorithm='backprop',stepmax=1e+5,startweight=weights,act.fct='logistic',learningrate=.01,hidden=c(2),linear.output=T,threshold=273.0633,lifesign.step=1,lifesign='full')
        /// print(nn)
        /// plot(nn)
        /// </summary>
        [TestMethod]
        public void NN_backpropagation_generic_std_one_hidden_pythagoras_single_data_0()
        {
            Init_dataset_pythagoras();
            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();
            build.SetParameters(0,1,.001,2500,.001);
            //build.SetNumberOfHiddenLayers(1);
            build.AddHiddenLayer(0, 2, new Sigmoid());
            build.SetOutputLayerActivationFunction(new Linear());

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);



            /*
            model.SetWeight(1, 0, 0, .2196);
            model.SetWeight(1, 1, 0, .121);
            model.SetWeight(1, 2, 0, -4.18);

            model.SetWeight(1, 0, 1, .15367);
            model.SetWeight(1, 1, 1, .2216);
            model.SetWeight(1, 2, 1, -.99404);

            model.SetWeight(2, 0, 0, 15.43);
            model.SetWeight(2, 1, 0, 12.92);
            model.SetWeight(2, 2, 0, -3.33);*/

            System.Diagnostics.Debug.WriteLine("Weight[1][0][0]:" + model.GetWeight(1, 0, 0));
            System.Diagnostics.Debug.WriteLine("Weight[1][1][0]:" + model.GetWeight(1, 1, 0));
            System.Diagnostics.Debug.WriteLine("Weight[1][2][0]:" + model.GetWeight(1, 2, 0));

            System.Diagnostics.Debug.WriteLine("Weight[1][0][1]:" + model.GetWeight(1, 0, 1));
            System.Diagnostics.Debug.WriteLine("Weight[1][1][1]:" + model.GetWeight(1, 1, 1));
            System.Diagnostics.Debug.WriteLine("Weight[1][2][1]:" + model.GetWeight(1, 2, 1));

            System.Diagnostics.Debug.WriteLine("Weight[2][0][1]:" + model.GetWeight(2, 0, 0));
            System.Diagnostics.Debug.WriteLine("Weight[2][1][1]:" + model.GetWeight(2, 1, 0));
            System.Diagnostics.Debug.WriteLine("Weight[2][2][1]:" + model.GetWeight(2, 2, 0));

            int row = 0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            System.Diagnostics.Debug.WriteLine("Final value:" + value);
            //Actual answer is 1.41
            Assert.IsTrue(value > 1.0 && value < 3.0);
                
        }

        [TestMethod]
        public void NN_backpropagation_generic_std_no_hidden_pythagoras_rmse()
        {
            Init_dataset_pythagoras();
            BuildGenericBackPropagationStandard build =
                    new BuildGenericBackPropagationStandard();
            build.SetParameters(0, 0,.3, 1);
            //build.SetNumberOfHiddenLayers(0);
            //build.AddHiddenLayer(0, 2, new Sigmoid());
            build.SetOutputLayerActivationFunction(new Linear());

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            double value = model.GetModelRMSE(_trainingData);

            Assert.IsTrue(value > 100000);
        }

        #endregion Regression (Mode=0) Tests

    }
}

using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.NeuralNetwork;
using Dasmic.Portable.Core;

namespace UnitTests.MLLib.Algorithms.NeuralNetwork
{
    [TestClass]
    public class _2LBackPropagationTest:BaseTest
    {
        #region Training Data from Jason Gaussian Naive Bayes

        [TestMethod]
        public void NN_backpropagation_2L_gnb_single_training_sample_class_0()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();

            build.SetParameters(1);
            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);
            
            int row = 0;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }

        [TestMethod]
        public void NN_backpropagation_2L_gnb_single_training_sample_class_1()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();
            build.SetParameters(1);

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int row = 5;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }

        
        [TestMethod]
        public void NN_backpropagation_2L_gnb_all_training_samples()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();
            build.SetParameters(1);

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

            Assert.AreEqual(count,
                10);
        }

        #endregion
       
        #region Training Data from Jason Naive Bayes

        [TestMethod]
        public void NN_backpropagation_2L_nb_sample_class_1()
        {
            initData_dataset_naive_bayes_jason_example();
            Build2LBackPropagation build =
                  new Build2LBackPropagation();
            build.SetParameters(1);

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            int row = 6;
            double[] data = GetSingleTrainingRowDataForTest(row);
            double value = model.RunModelForSingleData(data);

            Assert.AreEqual(value,
                _trainingData[_indexTargetAttribute][row]);
        }

        
        [TestMethod]
        public void NN_backpropagation_2L_nb_all_training_samples()
        {
            initData_dataset_naive_bayes_jason_example();
            Build2LBackPropagation build =
                  new Build2LBackPropagation();
            build.SetParameters(1);

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

        #region Custom Transfer Function
        [TestMethod]
        public void NN_backpropagation_2L_gnb_custom_activation_SigSig_all_training_samples()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();
            build.SetParameters(1);

            build.SetActivationFunction(0, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());
            build.SetActivationFunction(1, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());


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

        /// <summary>
        /// R:
        /// tdata=read.csv("C:\\Dasmic\\Development\\NET\\MLLib.NET\\DataSets\\pythagoras.csv",header=T)
        /// X = c(1, 2, 4, 3, 5)
        /// Y = c(1, 3, 3, 2, 5)
        /// tdata=as.matrix(cbind(X, Y))
        /// weights<-c(.005,.005,.005,.005,.005,.005,.005,.005,.005)
        /// library(neuralnet)
        /// nn <- neuralnet(Y ~ X,tdata, algorithm='backprop',stepmax=1e+5,startweight=weights,act.fct='logistic',hidden=c(2),learningrate=.3,linear.output=T,threshold=.01,lifesign.step=1,lifesign='full')
        /// </summary>
        [TestMethod]
        public void NN_backpropagation_2L_gnb_custom_activation_SigLin_all_training_samples()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();
            build.SetParameters(1,.01,4000,.05);

            build.SetActivationFunction(0, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());
            build.SetActivationFunction(1, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Linear());

        
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

            Assert.AreEqual(count,10);
        }

        [TestMethod]
        public void NN_backpropagation_2L_gnb_custom_activation_LinSig_all_training_samples()
        {
            initData_dataset_gaussian_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();
            build.SetParameters(1,.01,2000,.05);

            build.SetActivationFunction(0, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Linear());
            build.SetActivationFunction(1, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());

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
        public void NN_backpropagation_2L_nb_custom_activation_SigLin_all_training_samples()
        {            
            initData_dataset_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();
            build.SetParameters(1,.01,2000,.01);

            build.SetActivationFunction(0, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());
            build.SetActivationFunction(1, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Linear());

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

            Assert.AreEqual(count,
                8);
        }

        [TestMethod]
        public void NN_backpropagation_2L_nb_custom_activation_LinSig_all_training_samples()
        {
            initData_dataset_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();
            build.SetParameters(1);

            build.SetActivationFunction(0, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Linear());
            build.SetActivationFunction(1, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());
            
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

            Assert.AreEqual(count,
                8);
        }

        [TestMethod]
        public void NN_backpropagation_2L_nb_custom_activation_SigTanh_all_training_samples()
        {
            initData_dataset_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();
            build.SetParameters(1);

            build.SetActivationFunction(0, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());
            build.SetActivationFunction(1, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.HyperbolicTangent());

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

            Assert.AreEqual(count,
                8);
        }

        [TestMethod]
        public void NN_backpropagation_2L_nb_custom_activation_SigSig_all_training_samples()
        {
            initData_dataset_naive_bayes_jason_example();
            Build2LBackPropagation build =
                    new Build2LBackPropagation();
            build.SetParameters(1);

            build.SetActivationFunction(0, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());
            build.SetActivationFunction(1, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());

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

        #region Regression Test

        /// <summary>
        /// R:
        /// tdata=read.csv("C:\\Dasmic\\Development\\NET\\MLLib.NET\\DataSets\\pythagoras.csv",header=T)
        /// X = c(1, 2, 4, 3, 5)
        /// Y = c(1, 3, 3, 2, 5)
        /// tdata=as.matrix(cbind(X, Y))
        /// weights<-c(.005,.005,.005,.005,.005,.005,.005,.005,.005)
        /// library(neuralnet)
        /// nn <- neuralnet(Y ~ X,tdata, algorithm='backprop',stepmax=1e+5,startweight=weights,act.fct='logistic',hidden=c(2),learningrate=.3,linear.output=T,threshold=.01,lifesign.step=1,lifesign='full')
        /// </summary>
        [TestMethod]
        public void NN_backpropagation_2L_jason_simple_rmse()
        {
            Init_dataset_jason_linear_regression();
            Build2LBackPropagation build =
                 new Build2LBackPropagation();
            build.SetParameters(0,.01,3000,.05);
            build.SetActivationFunction(0, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Sigmoid());
            build.SetActivationFunction(1, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Linear());

            ModelBackPropagationBase model =
                (ModelBackPropagationBase)build.BuildModel(
                    _trainingData, _attributeHeaders,
                    _indexTargetAttribute);

            double value = model.GetModelRMSE(_trainingData);

            Assert.IsTrue(value < 1.0);
        }

        #endregion

    }
}

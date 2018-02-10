using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    /// <summary>
    /// Build a Feed-Forward Neural Network using multiple output neurons,
    /// solved using Stoicastic Gradient Descent (SGD). This algorithm is a modification
    /// of the Perceptron algorithm and is meant for multi-classification problems only.
    /// There is no hidden layer in this Perceptron network
    /// 
    /// This is a single layer and multi neuron network with Perceptron function
    /// The algorithm automatically sets the number of neurons to the number of classes
    /// available in training data
    /// </summary>
    public class BuildMultiUnitPerceptronSGD:BuildPerceptronBase
    {               
        public override Common.MLCore.ModelBase 
            BuildModel(double[][] trainingData,
                         string[] attributeHeaders,
                         int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);
            ModelMultiUnitPerceptron model = 
                new ModelMultiUnitPerceptron(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1);

            //Find out number of categories
            //Get unique value in Target Attribute
            HashSet<double> uniqueTargetValues =
                    GetUniqueValues(trainingData[indexTargetAttribute]);
            model.TargetValues = uniqueTargetValues.ToArray<double>();

            Support.ActivationFunction.IActivationFunction actFunction =
               new Support.ActivationFunction.Linear();
            model.SetNumberOfLayers(1);
            model.SetInputLayer(_trainingData, _noOfAttributes);
            model.AddLayer(1, uniqueTargetValues.Count, 
                    _weightBaseValue, actFunction, _maxParallelThreads);
            
            double _learningRate=0;
                     
            //Start the training
            //Do not parallelize as computation has to be sequential
            for (int epoch = 0; epoch < _noOfEpoch; epoch++) //Do not parallelize
            {
                _learningRate = _alpha * (1 - (epoch / 
                                        _noOfEpoch));                   
                //Cannot be parallelized, has to be a sequential operation
                for (long row = 0; row < _noOfDataSamples; row++) //For each value in trainingData row
                {
                    //Find max Idx in computedValue
                    int maxIdx = model.GetIndexOfMaxOutput(row);

                    if (model.TargetValues[maxIdx] != _trainingData[_indexTargetAttribute][row]) //InCorrect prediction
                    {
                        Parallel.For(0, _noOfAttributes,
                            new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, wIdx =>
                        //for (int wIdx = 0; wIdx < _noOfAttributes; wIdx++)
                        {
                            double mulValue=-1;
                            if (model.TargetValues[wIdx] == _trainingData[_indexTargetAttribute][row]) //Is Correct Row
                                mulValue = 1;
                            
                            for (int col=0;col<_noOfAttributes; col++)
                            {
                                model.SetWeightOutputLayer(col,wIdx,
                                    model.GetWeightOutputLayer(col,wIdx) + mulValue * _learningRate * _trainingData[col][row]);
                            }
                            //For Intercept
                            model.SetBiasOutputLayer(wIdx,
                                    model.GetBiasOutputLayer(wIdx) + mulValue * _learningRate * 1.0);

                        });
                    }                  
                } //training row                
            } //epoch            
            return model;            
        } //build model function        
    } //class
}

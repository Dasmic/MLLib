using System.Threading.Tasks;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    
    /// <summary>
    /// Build a Feed-Forward Neural Network using Single neuron Perceptron,
    /// solved using Stoicastic Gradient Descent (SGD). The Perceptron
    /// function can take values of 0 or 1 and it suited only for classification
    /// problems. There is no hidden layer in this Perceptron network
    /// 
    /// This is a single layer and single neuron network with Perceptron function
    /// who's value are binary. The input data should also have only
    /// two classes
    /// </summary>
    public class BuildSingleUnitPerceptronSGD:BuildPerceptronBase
    {        
        /// <summary>
        /// 0 - Alpha (Initial Learning Rate); default=.3
        /// 1 - Maximum Epoch/Iterations;default=100
        /// </summary>
        /// <param name="values"></param>
        public override void
            SetParameters(params double[] values)
        {
            if (values.Length > 0)
                if (values[0] != double.NaN)
                    _alpha = (int)values[0];
            if (values.Length > 1)
                if (values[1] != double.NaN)
                    _noOfEpoch = (int)values[1];
        }

        public override Common.MLCore.ModelBase 
            BuildModel(double[][] trainingData,
                         string[] attributeHeaders,
                         int indexTargetAttribute)
        {
            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);
            ModelSingleUnitPerceptron model = 
                new ModelSingleUnitPerceptron(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1);

            //Init Primary Weights to 0
            Support.ActivationFunction.IActivationFunction actFunction =
                new Support.ActivationFunction.Step();
            model.SetNumberOfLayers(1);
            model.SetInputLayer(_trainingData, _noOfAttributes);
            model.AddLayer(1,1, _weightBaseValue, actFunction, _maxParallelThreads);
            
            double _learningRate=0;
            //Start the training
            //Do not parallelize as computation has to be sequential
            for (int epoch = 0; epoch < _noOfEpoch; epoch++) //Do not parallelize
            {
                _learningRate = _alpha * (1 - (epoch / 
                                        _noOfEpoch));                
                double computedValue = 0;
                
                //Cannot be parallelized, has to be a sequential operation
                for (long row = 0; row < _noOfDataSamples; row++) //For each value in trainingData row
                {                    
                    //Get current value Reuse existing function in model
                    computedValue = model.GetOutput(0,row);
                    //Now update weights - Stoichastic Gradient Descent
                    for (int col = 0; col < _noOfAttributes; col++)
                    //Parallel.For(0, _noOfAttributes,
                    //    new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, col =>
                    {
                        //double weight = model.GetWeightOutput(col);
                        //weight +=
                        //        _learningRate * (_trainingData[_indexTargetAttribute][row] -
                        //            computedValue) * _trainingData[col][row];
                        model.SetWeightOutputLayer(col,0, model.GetWeightOutputLayer(col,0)+
                                    _learningRate * (_trainingData[_indexTargetAttribute][row] -
                                    computedValue) * _trainingData[col][row]);
                    }//);
                    //For Intercept
                    model.SetBiasOutputLayer(0,model.GetBiasOutputLayer(0) +
                            _learningRate * (_trainingData[_indexTargetAttribute][row] -
                                        computedValue) * 1.0);                    
                } //training row                
            } //epoch
            return model;
        } //build model function
    } //class
}

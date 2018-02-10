using System;
using System.Threading.Tasks;
using System.Linq;
using System.Collections.Generic;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    public abstract class BuildBackPropagationBase:BuildBase
    {
        protected ModelBackPropagationBase.EnumMode _mode;
        protected int _noOfUnitsOutputLayer;
        protected IActivationFunction[] _activationFunctions;
        protected ModelBackPropagationBase _model;

        public abstract override Common.MLCore.ModelBase
                 BuildModel(double[][] trainingData,
                 string[] attributeHeaders,
                 int indexTargetAttribute);

        /// <summary>
        /// Add Hidden Layer, starting from index 0 
        /// Do not add Output (idxLayer=numberOfLayers) Layers
        /// 
        /// Output Layer will be determined automatically based
        /// on where Neural Net is running in classification of Regression mode
        /// </summary>
        /// <param name="idxLayer"></param>
        /// <param name="numberOfUnits"></param>
        /// <param name="activationFunction"></param>
        public void AddHiddenLayer(int idxLayer,
                        int numberOfUnits,
                        IActivationFunction activationFunction)
        {
            if (_model.GetNumberOfLayers() == 0)
                throw new InvalidNeuralNetworkLayer();

            //Input layer is already added by now
            if (idxLayer + 1 >= _model.GetNumberOfLayers() - 1) //throw Error
                throw new InvalidNeuralNetworkLayer();


            _model.AddLayer(idxLayer + 1, numberOfUnits, //Hidden Lalyer
                   _weightBaseValue,
                   activationFunction,
                   _maxParallelThreads);
        }

        public double[] GetNumberOfTargetValues(ModelBackPropagationBase.EnumMode mode,
                                            double [][]trainingData,
                                            int indexTargetAttribute)
        {

            if (_mode == ModelBackPropagationBase.EnumMode.Classification) //Classification
            {
                HashSet<double> uniqueTargetValues =
                        GetUniqueValues(trainingData[indexTargetAttribute]);
                return uniqueTargetValues.ToArray<double>();
            }
            else //Regression
                return new double[1]; //Returns a single array as only array count is required in regression
        }
        /// <summary>
        /// Returns combined threshold value
        /// </summary>
        /// <param name="computedValuesAllLayers"></param>
        /// <param name="errorLayers"></param>
        /// <param name="idxOutputLayer"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        protected double ComputeError(double[][] computedValuesAllLayers,
                                    double[][] errorLayers,
                                    int idxOutputLayer, long row)
        {
            double sumError = 0;
            object mutex = new object();
            //For each output unit, can go in parallel - Compute errors for output unit 
            Parallel.For(0, computedValuesAllLayers[idxOutputLayer].Length,
                    new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                    idxOutputUnit =>
                    //for (int idxOutputUnit = 0; idxOutputUnit < computedValuesAllLayers[idxOutputLayer].Length; idxOutputUnit++)
                    {
                        double expectedOutput = 0;

                        computedValuesAllLayers[idxOutputLayer][idxOutputUnit] =
                                        _model.GetOutput(idxOutputUnit, row);

                        //Will depend on mode
                        //Number of units in outputs =  model.TargetValues
                        if (_mode == ModelBackPropagationBase.EnumMode.Classification) //Classification
                        {
                            //expectedOutput = 1 if correct output unit
                            //expectedOutput = 0 if incorrect output unit
                            if (_trainingData[_indexTargetAttribute][row]
                                        == _model.TargetValues[idxOutputUnit])
                                expectedOutput = 1;
                            else
                                expectedOutput = 0;
                        }
                        else //for regression
                            expectedOutput = _trainingData[_indexTargetAttribute][row];


                        errorLayers[idxOutputLayer][idxOutputUnit] = _model.GetDerivativeValue(idxOutputLayer,
                                                    computedValuesAllLayers[idxOutputLayer][idxOutputUnit]) *
                                                    GetOutputUnitDifference(computedValuesAllLayers[idxOutputLayer][idxOutputUnit], expectedOutput);                                                    
                        lock (mutex)
                        {
                            sumError += Math.Abs(errorLayers[idxOutputLayer][idxOutputUnit]);
                        }
                    });//idxOutput

            //Compute errors for all hidden unit 
            //Start from right most  to left most
            for (int idxHiddenLayer = idxOutputLayer - 1; idxHiddenLayer > 0; idxHiddenLayer--)
            {
                Parallel.For(0, computedValuesAllLayers[idxHiddenLayer].Length,
                        new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                        idxHiddenUnit =>
                        //for (int idxHiddenUnit = 0; idxHiddenUnit < computedValuesAllLayers[idxHiddenLayer].Length;
                        //                           idxHiddenUnit++)
                        {
                            computedValuesAllLayers[idxHiddenLayer][idxHiddenUnit] =
                                _model.GetOutputTillLayer(idxHiddenLayer, idxHiddenUnit, row);

                            double sumHidden = 0;
                            //Compute sum of output unit
                            for (int idxUnitRightLayer = 0; idxUnitRightLayer < computedValuesAllLayers[idxHiddenLayer + 1].Length; idxUnitRightLayer++)
                            {
                                sumHidden +=
                                           errorLayers[idxHiddenLayer + 1][idxUnitRightLayer] * //Use error of output layer
                                               _model.GetWeight(idxHiddenLayer + 1, idxHiddenUnit, idxUnitRightLayer); //+2 is the actual downstream layer idx
                            }

                            errorLayers[idxHiddenLayer][idxHiddenUnit] =
                                 _model.GetDerivativeValue(idxHiddenLayer,
                                                            computedValuesAllLayers[idxHiddenLayer][idxHiddenUnit]) *
                                                            (sumHidden);
                        });
            } //idxHiddenLayer
            return sumError;
        }

        /// <summary>
        /// This function should be reimplemented for any routine
        /// that is following the standard backpropagation as given in Mitchell's book
        /// </summary>
        /// <param name="computedValuesAllLayers"></param>
        /// <param name="expectedOutput"></param>
        /// <returns></returns>
        protected virtual double GetOutputUnitDifference(double computedValuesAllLayers, double expectedOutput)
        {
            //returns tk- ok
            return expectedOutput - computedValuesAllLayers;
        }
        
        

        protected bool VerifyUpstreamLayers(ModelBackPropagationBase model)
        {
            bool exceptionFlag = false;
            Parallel.For(1, model.GetNumberOfLayers(),
                   new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                   idxLayer =>
                   {
                       if (model.GetLayer(idxLayer).GetNumberOfUpstreamUnits() < 0)//Check if Upstream Layer is setup, this will set an exception otherwise
                           exceptionFlag = true;
                   });
            return exceptionFlag;
        }

        protected bool VerifyLayers(ModelBackPropagationBase model)
        {
            bool exceptionFlag = false;
            Parallel.For(0, model.GetNumberOfLayers() - 1,
                   new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                   idxLayer =>
                   {
                       if (model.GetLayer(idxLayer) == null)
                           exceptionFlag = true;
                   });
            return exceptionFlag;
        }


    }
}

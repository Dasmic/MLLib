using System.Collections.Generic;
using System.Threading.Tasks;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    public abstract class ModelBase : MLLib.Common.MLCore.ModelBase
    {
        //Should be able to take single or multiple weights
        //Multiple weights are for multi-class problems
        private SingleLayer[] _layers { get; set; }

        public ModelBase(double missingValue,
                                int indexTargetAttribute, int countAttributes) :
                                base(missingValue, indexTargetAttribute, countAttributes)
        {
            _layers = new SingleLayer[1 + 1]; //Set 1 hidden layer
        }

        public SingleLayer GetLayer(int idxLayer)
        {
            return _layers[idxLayer];
        }


        /// <summary>
        /// Do not add Input Layers, it is set by default
        /// 
        /// </summary>
        /// <param name="layerCount"></param>
        public void SetNumberOfLayers(int layerCount)
        {
            _layers = new SingleLayer[layerCount + 1]; //1 for output layer            
        }

        public void SetInputLayer(double[][] inputData, int origAttributeCount)
        {
            _layers[0] = new SingleLayerInput(origAttributeCount, _maxParallelThreads);
            ChangeInputLayerData(inputData);
        }

        public void ChangeInputLayerData(double[][] inputData)
        {            
            if (_layers == null)
                throw new IndexOutOfRangeException();

            ((SingleLayerInput)(_layers[0])).SetData(inputData);
        }

        public void AddLayer(int layerIdx, long numberOfUnits, double weightInitValue,
                    IActivationFunction actFunction,int maxParallelThreads)
        {
            if (layerIdx > _layers.Length - 1)
                throw new IndexOutOfRangeException();
            int upStreamIdx=0;

            if (layerIdx > 0)
                upStreamIdx = layerIdx-1;

            _layers[layerIdx] = new SingleLayer(numberOfUnits, weightInitValue,
                _layers[upStreamIdx], actFunction, _maxParallelThreads);
        }

        public double GetWeight(int idxLayer,long idxUpUnit, long idxUnit)
        {
            return _layers[idxLayer].Weights[idxUpUnit][idxUnit];
        }

        public void SetWeight(int idxLayer, long idxUpUnit, long idxUnit, double value)
        {
            _layers[idxLayer].Weights[idxUpUnit][idxUnit] = value;
        }


        public long GetNumberOfUnits(int idxLayer)
        {
            return _layers[idxLayer].GetNumberOfUnits();
        }

        /// <summary>
        /// Does not include bias term
        /// </summary>
        /// <param name="idxLayer"></param>
        /// <returns></returns>
        public long GetNumberOfUpstreamUnits(int idxLayer)
        {
            return _layers[idxLayer].GetNumberOfUpstreamUnits();
        }

        public double GetUpstreamUnitValue(int idxLayer, long idxUpUnit,long row)
        {
            return _layers[idxLayer].GetValueUpstreamLayer(idxUpUnit,row);
        }

        public double GetUnitValue(int idxLayer, long idxUnit, long row)
        {
            return _layers[idxLayer].GetValue(idxUnit, row);
        }

        public double[] GetOutput(long row)
        {
            double[] computedValues =
                GetNeuralNetworkOutputForSingleDataRow(row);

            return computedValues;
        }

        public double GetOutputTillLayer(int idxUptillLayer, long idxUnit,long row)
        {
            double computedValue =
                GetNeuralNetworkOutputSingleDataRowSingleUnit(idxUptillLayer, 
                        idxUnit,row);

            return computedValue;
        }

        public double GetOutput(long unitIdx, long row)
        {
            double[] computedValues =
                GetOutput(row);

            if (computedValues != null)
                return computedValues[unitIdx];
            else
                return _missingValue;
        }


        /// <summary>
        /// Computes the output of the full Neural Network
        /// </summary>
        /// <param name="dataRow"></param>
        /// <returns></returns>
        public double[] GetNeuralNetworkOutputForSingleDataRow(long dataRow)
        {
            int idxUptillLayer = _layers.Length - 1;
            return GetNeuralNetworkOutputForSingleDataRow(idxUptillLayer,dataRow);
        }

        public double[] GetNeuralNetworkOutputForSingleDataRow(int idxUptillLayer,
                            long dataRow)
        {
            
            SingleLayer lastLayer = _layers[idxUptillLayer];
            long lastLayerNoOfUnits = lastLayer.GetNumberOfUnits();
            double[] outputs = new double[lastLayerNoOfUnits];
            
            //Only run for last layer
            for (int uIdx = 0; uIdx < lastLayerNoOfUnits; uIdx++)
            {
                outputs[uIdx] = 
                    GetNeuralNetworkOutputSingleDataRowSingleUnit(idxUptillLayer,
                                        uIdx, dataRow);
            }

            return outputs;
        }


        /// <summary>
        /// Returns value of single unit in a single layer for single data row
        /// </summary>
        /// <param name="idxUptillLayer"></param>
        /// <param name="idxUnit"></param>
        /// <param name="dataRow"></param>
        /// <returns></returns>
        public double GetNeuralNetworkOutputSingleDataRowSingleUnit(
                        int idxUptillLayer, 
                        long idxUnit,
                       long dataRow)
        {
            //((SingleLayerInput)(_layers[0])).SetCurrentRow(dataRow);
            SingleLayer lastLayer = _layers[idxUptillLayer];

            return lastLayer.GetValue(idxUnit,dataRow);
         }

            
        /// <summary>
        /// Gets number of layers including input layer
        /// </summary>
        /// <returns></returns>
        public int GetNumberOfLayers()
        {
            if (_layers != null)
                return _layers.Length;
            else
                return 0;
        }

        /// <summary>
        /// Gets number of hidden layers
        /// </summary>
        /// <returns></returns>
        public int GetNumberOfHiddenLayers()
        {
            if (_layers != null)
                return _layers.Length - 2;
            else
                return 0;
        }



        public double GetDerivativeValue(int idxLayer, double value)
        {
            return _layers[idxLayer].GetDerivativeValue(value);
        }

        public override abstract
           double RunModelForSingleData(double[] data);   
           
        //Serialization Routine
        public override void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }

        

    }
}

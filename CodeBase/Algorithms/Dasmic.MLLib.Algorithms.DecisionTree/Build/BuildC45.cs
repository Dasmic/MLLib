using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Math.Statistics;
using System.Collections.Concurrent;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public class BuildC45 : BuildBase
    {
        

        /// <summary>
        /// Values format is Col, Row
        /// </summary>
        /// <param name="values"></param>
        /// <param name="headers"></param>
        public override Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute)
        {

            //Verify data and set variables
            VerifyData(trainingData, attributeHeaders, indexTargetAttribute);

            ModelC45 mb = new ModelC45(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1);

            //Data is good, proceed
            //Set row indices before recursion
            ConcurrentBag<long> trainingDataRowIndices = new ConcurrentBag<long>();
            int length = _trainingData[0].Length;
            Parallel.For(0, length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, rowIdx =>
            {
                trainingDataRowIndices.Add(rowIdx);
            });

            mb.Root = buildChildNodes(trainingDataRowIndices, 0, null);
            //Prune the Tree
            return mb;
        }

        protected void pruneTree()
        {
            //if (_root == null) return;
        }
        
        /// <summary>
        /// Gets data with lowest Entropy (Highest Information Gain)
        /// 
        /// Checks for Missing Values and Ignores them
        /// </summary>
        /// <param name="filteredData"></param>
        /// <returns>HighestEntropyData object. Will be null if TargetAttributes</returns>
        protected override SplittedAttributeData
                splitDataOnUnivariateCriterion(
                        double[][] data)
        {
            double entropyS;
            double entropySv;
            double entropySum;
            double infoGain = 0;
            double[] filteredTargetData;
            Dictionary<double, long> freqs;
            SplittedAttributeData ed = new SplittedAttributeData();

            ed.SplittingCriteriaValue = double.NegativeInfinity;
            entropyS = getEntropyOfTargetAttribute(data);

            for (int idxCol = 0; idxCol < data.Count();
                                    idxCol++)
            {
                entropySum = 0;
                if (data[idxCol] != null &&
                            idxCol != _indexTargetAttribute) //Do not compute when data not present
                {
                    freqs = InformationGain.Frequency(
                                data[idxCol]);
                    //key has value
                    foreach (double key in freqs.Keys)
                    {
                        if(key !=_missingValue)
                        { 
                            filteredTargetData =
                                getFilteredTargetValues(data, idxCol, key);

                            entropySv = InformationGain.EntropyShannon(filteredTargetData);
                            entropySum += ((double)filteredTargetData.Length /
                                            (double)data[_indexTargetAttribute].Length)
                                            * entropySv;
                        }
                    }

                    infoGain = entropyS - entropySum;
                    //Compute InfoGain   
                    if (infoGain > ed.SplittingCriteriaValue)
                    {
                        ed.Freqs = freqs;
                        ed.AttributeIndex = idxCol;
                        ed.SplittingCriteriaValue = infoGain;
                    }
                }//if condition
            } //Main loop

            return ed;
        }


    } //Class
} //Namespace

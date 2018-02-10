using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Math.Statistics;
using Dasmic.MLLib.Common.MLCore;
using System.Collections.Concurrent;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public class BuildID3:BuildBase
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
            VerifyData(trainingData,attributeHeaders,indexTargetAttribute);

            ModelID3 mb = new ModelID3(_missingValue,
                                                _indexTargetAttribute,
                                                _trainingData.Length - 1);

            //Data is good, proceed
            //Set row indices before recursion
            ConcurrentBag<long> trainingDataRowIndices = new ConcurrentBag<long>();
            int length = _trainingData[0].Length;
            
            //for (int rowIdx=0;rowIdx<length;rowIdx++)            
            Parallel.For(0, length, new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },rowIdx =>
            {
                //Thread safe insert
                trainingDataRowIndices.Add(rowIdx);
            });

            //trainingDataRowIndices is used to track tree values
            mb.Root=buildChildNodes(trainingDataRowIndices,0,null);
            return mb;
        }
        
        
        /// <summary>
        /// Gets data with lowest Entropy (Highest Information Gain)
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
            double infoGain=0;
            double[] filteredTargetData;
            Dictionary<double, long> freqs;
            SplittedAttributeData ed=new SplittedAttributeData();

            ed.SplittingCriteriaValue = double.NegativeInfinity;
            entropyS = getEntropyOfTargetAttribute(data);

            for (int ii=0;ii< data.Count(); 
                                    ii++)
            {
                entropySum = 0;
                if (data[ii] != null &&
                            ii != _indexTargetAttribute) //Do not compute when data not present
                { 
                        freqs = InformationGain.Frequency(
                                    data[ii]);
                        foreach (double key in freqs.Keys)
                        {
                            filteredTargetData =
                                    getFilteredTargetValues(data, ii, key);

                            entropySv = InformationGain.EntropyShannon(filteredTargetData);
                            entropySum += ((double)filteredTargetData.Length /(double) data[_indexTargetAttribute].Length) 
                                                *entropySv;
                            
                        }
                        infoGain = entropyS - entropySum;
                    //Compute InfoGain   
                    if (infoGain > ed.SplittingCriteriaValue)
                    {
                        ed.Freqs = freqs;
                        ed.AttributeIndex = ii;
                        ed.SplittingCriteriaValue = infoGain;
                    }
                }//If condition
                 
            } //Main loop
           
            return ed;
        }

        

    } //Class
} //Namespace

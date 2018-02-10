using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;


namespace Dasmic.MLLib.Common.IO.Windows
{
    public class CSVOperations: IFileOperations
    {                
        public void Write(FileData fd, 
                    string fullFilePath)
        {
            String lineS="";
            
            TextFileWriter fw = new TextFileWriter(
                fullFilePath);

            //Write Headers
            for (long col = 0; col < fd.attributeHeaders.Length; col++)
            {
                lineS = lineS + fd.attributeHeaders[col];
                if (col < fd.attributeHeaders.Length - 1)
                    lineS = lineS + ",";
            }
            fw.WriteLine(lineS);

            //Write Data
            for (long row=0; row < fd.values[0].Length; row++)
            {
                lineS = "";
                for (long col=0; col < fd.attributeHeaders.Length; col++)
                {
                    lineS = lineS + fd.values[col][row];
                    if (col < fd.attributeHeaders.Length - 1)
                        lineS = lineS +  ",";
                }
                fw.WriteLine(lineS);
            }
            fw.CompleteWrite();
        }

        public void Write(double[][] values, 
                                        string[] attributeHeaders,
                                            string fullFilePath)
        {
            FileData fd;
            fd.attributeHeaders = attributeHeaders;
            fd.values = values;

            Write(fd, fullFilePath);
        }


        /// <summary>
        /// File should be in format:
        /// 
        /// X1,X2,Y
        /// 1,4,6.2
        /// 3,8,4.5
        /// ...
        /// 
        /// Value should be in right most column
        /// NOTE: Sequence of data is not guaranteed
        /// 
        /// # can be used as comment
        /// </summary>
        /// <param name="filePath"></param>
        /// <param name="maxParallelThreads">-1 if unlimited threads</param>
        /// <returns></returns>
        public FileData Read(string fullFilePath,
                                int maxParallelThreads)
        {
            FileData fd = new FileData();
            TextFileReader tr = new TextFileReader(fullFilePath);

            //Read first line - will have attribute headers
            string line="#";
            while (line.StartsWith("#"))
                line = tr.ReadLine();
            fd.attributeHeaders = line.Split(',');

            if (fd.attributeHeaders.Length < 1)
                throw new InvalidDataSetFileException();

            fd.values = new double[fd.attributeHeaders.Length][];
            ConcurrentDictionary<long,string> allLines =
                new ConcurrentDictionary<long, string>();

            int idxLine = 0;
            while (line != null)
            {
                line = tr.ReadLine();
                if (line != null)
                    if(!line.StartsWith("#")) //Remove comments
                        allLines.AddOrUpdate(idxLine++,line,(key,oldValue)=>oldValue);
            }
            tr.CompleteRead();

            //Set number of rows in each column
            int count = allLines.Count;
            Parallel.For(0, fd.values.Length,
                   new ParallelOptions { MaxDegreeOfParallelism = maxParallelThreads },
                   ii =>
                //for (int ii=0;ii< fd.values.Length; ii++)
                {
                    fd.values[ii] = new double[count];
                });

            //Now start parsing
            bool flag = true;
            
            Parallel.For(0, allLines.Count,
                   new ParallelOptions { MaxDegreeOfParallelism = maxParallelThreads },
                   row =>
            //for (int row = 0; row < allLines.Count; row++)
            {
                if (allLines != null)
                {
                    string value = allLines[row];
                    string[] values = value.Split(',');

                    if (values.Length != fd.values.Length) //Corrupt
                        flag = false;
                    for (int idx = 0; idx < values.Length; idx++)
                    {
                        if (values[idx] == null)
                            flag = false;
                        if (values[idx].Trim() == "") //Every value should be good
                            flag = false;
                        double.TryParse(values[idx], out fd.values[idx][row]);
                    }
                }
            });

            if(!flag) //Some line was corrupted
                throw new InvalidDataSetFileException();

            return fd;
        }
    }
}

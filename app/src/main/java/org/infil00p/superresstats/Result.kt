package org.infil00p.superresstats

import java.util.*

class Result(val framework: String, val duration: Double, val imageUri: String, val device: String ) {

}

class ResultSet(val resultList: Vector<Result>)
{
    fun calculateAverage() : Double
    {
        var total = 0.0;
        for(result in resultList) {
            total += result.duration;
        }
        return total/resultList.size;
    }

    fun getMin() : Double
    {
        var lowestNum = resultList.firstElement().duration;
        for(result in resultList){
            if(lowestNum > result.duration)
                lowestNum = result.duration
        }
        return lowestNum
    }

    fun getMax() : Double
    {
        var highestNum = resultList.firstElement().duration;
        for(result in resultList){
            if(highestNum < result.duration)
                highestNum = result.duration
        }
        return highestNum
    }

}
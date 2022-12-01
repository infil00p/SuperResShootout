/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 ~ Copyright 2022 Adobe
 ~
 ~ Licensed under the Apache License, Version 2.0 (the "License");
 ~ you may not use this file except in compliance with the License.
 ~ You may obtain a copy of the License at
 ~
 ~     http://www.apache.org/licenses/LICENSE-2.0
 ~
 ~ Unless required by applicable law or agreed to in writing, software
 ~ distributed under the License is distributed on an "AS IS" BASIS,
 ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 ~ See the License for the specific language governing permissions and
 ~ limitations under the License.
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

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
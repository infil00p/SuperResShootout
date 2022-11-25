package org.infil00p.superresstats

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream

class AssetHandler internal constructor(var mCtx: Context) {
    var LOGTAG = "AssetHandler"

    inner class ModelFileInit internal constructor(
        var mModelName: String,
        var mDataDir: File,
        var mAssetManager: AssetManager,
        var mModelFiles: Array<String>
    ) {
        var mTopLevelFolder: File? = null
        var mModelFolder: File? = null

        @Throws(IOException::class)
        private fun InitModelFiles() {
            copyModelFiles()
        }

        @Throws(IOException::class)
        private fun copyFileUtil(
            files: Array<String>
        ) {
            // For this example, we're using the internal storage
            for (file in files) {
                val inputFile = mAssetManager.open("$mModelName/$file")
                var outFile: File
                val dir = File(mDataDir.toString() + "/" +  mModelName)
                outFile = File(dir, file)
                val out: OutputStream = FileOutputStream(outFile)
                val buffer = ByteArray(1024)
                var length: Int
                while (inputFile.read(buffer).also { length = it } != -1) {
                    out.write(buffer, 0, length)
                }
                inputFile.close()
                out.flush()
                out.close()
            }
        }

        @Throws(IOException::class)
        private fun copyModelFiles() {
            copyFileUtil(mModelFiles)
        }


        private fun createTopLevelDir() {
            mTopLevelFolder = File(mDataDir.absolutePath, mModelName)
            mTopLevelFolder!!.mkdir()
        }

        init {
            createTopLevelDir()
            InitModelFiles()
        }
    }

    @Throws(IOException::class)
    private fun Init() {
        val dataDirectory = mCtx.filesDir
        val assetManager = mCtx.assets

        val tflite = ModelFileInit(
            "tflite",
            dataDirectory,
            assetManager,
            arrayOf(
                "model_float32.tflite"
            )
        )

        val pyTorch = ModelFileInit(
            "pytorch",
            dataDirectory,
            assetManager,
            arrayOf(
                "superres.pt",
                "superres_nhwc.pt"
            )
        )

        val ort = ModelFileInit(
            "ort",
            dataDirectory,
            assetManager,
            arrayOf(
                "super_resolution.with_runtime_opt.ort"
            )
        )

        // This is pretty terrible, to be honest, and we could do better
        val image_set = ModelFileInit(
            "image_set",
            dataDirectory,
            assetManager,
            arrayOf(
                "01.png","02.png","03.png","04.png","05.png","06.png","07.png","08.png","09.png",
                "10.png", "11.png","12.png","13.png","14.png","15.png","16.png","17.png","18.png",
                "19.png", "20.png","21.png","22.png","23.png","24.png","25.png"
            )
        )

    }

    init {
        try {
            Init()
        } catch (e: IOException) {
            Log.d(LOGTAG, "Unable to get models from storage")
        }
    }
}
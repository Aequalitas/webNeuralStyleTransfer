// Leon Gaty`s neural style transfer with Tensorflow.js
function handleFileSelect(evt){
  let reader = new FileReader();
  reader.onload = file => {
    modelTexturePath = file.target.result;
    document.getElementById( evt.target.getAttribute('imagetype')+'Image').src = file.target.result;
  };

  reader.onprogress =  xhr => {
    if (xhr.lengthComputable){
      let percentComplete = Math.round(xhr.loaded / xhr.total * 100, 2);
      console.log(percentComplete + '% loaded of image');
    }
  }

  reader.onerror = xhr => {
        alert('An error occurred reading this file.');
        console.log(xhr.target.error);
      }

  reader.readAsDataURL(evt.target.files[0]);
}


	let canvas = document.getElementById('canvas');
	canvas.width = canvas.parentElement.offsetWidth;
 	canvas.height = canvas.parentElement.offsetHeight;

	canvasContext = canvas.getContext('2d');

	make_base();

	function make_base()
	{
	  base_image = new Image();
	  base_image.src = 'img/example.png';
	  base_image.onload = function(){
	    canvasContext.drawImage(base_image, 0, 0, canvas.width, canvas.height);
	  }
	}


      let model = null;
      let IMAGEDIM = $("#imagedim").val();
      let MAXSTEPS = $("#steps").val();
      let LEARNRATE = $("#tfrate").val();
      let STYLEWEIGHT = $("#styleweight").val();
      // normalizing vector for the VGG19 model
      const normV = tf.tensor1d([100.939, 116.779, 123.68])


      console.log("Loading Model...")
      // first 5 are the style layers and the last is the content layer
      model = tf.loadModel("/models/VGG19/model.json").then(loadedModel);

      function loadedModel(_model) 
      {

   
        model = _model;

	//tf.setBackend("cpu");
        console.log("Model loaded! Current Backend: "+tf.getBackend());
	$("#startTransforming").text("Start transforming");
	$("#startTransforming").click(initTransform);
	$("#startTransforming").removeClass("disabled");

      }

      function getImage(type){

        let image = tf.fromPixels(document.getElementById(type+"Image"), 3).asType("float32")
        image = tf.image.resizeNearestNeighbor(image, [IMAGEDIM, IMAGEDIM])
        image = tf.expandDims(image)//.add(normV)
        //console.log(type+" image: "+image.shape)
	//image = image.sub(normV);
        return image;
      }

      function calcStyleLoss(weightsFeatures, features){
		return tf.tidy( ()  => {

       		// style loss
     		let weightsGram = gramMatrix(weightsFeatures[0]);
      		let styleLoss = tf.losses.meanSquaredError(features[1][0], weightsGram);
		//console.log("style loss: ("+weightsGram.shape+ " - "+ features[1][0].shape+" ) => "+styleLoss)

	        for(let s=1; s < 4; s++){
       			weightsGram = gramMatrix(weightsFeatures[s]);
      			//console.log("style loss:  ("+weightsGram.shape+ " - "+ features[1][s].shape+" )")
     			styleLoss = tf.losses.meanSquaredError(weightsGram, features[1][s]).add(styleLoss).div(2);
   		}

		return styleLoss;
		});

	}

      function calcContentLoss(weightsFeatures, features) {
		 return tf.tidy( ()  => {
          		//console.log(weightsFeatures[4].dataSync());
          		// content loss
          		let weightsContent = weightsFeatures[4];
          		let contentLoss = tf.losses.meanSquaredError(weightsContent, features[0]);
	  		//console.log("content loss: ("+weightsContent.shape+ " - "+ features[0].shape+") => "+contentLoss)

			return contentLoss;
      		});
	};

      function gramMatrix(input){
	  return tf.tidy( ()  => {
          	let n = tf.scalar(input.shape[0]).asType("float32")
          	input = tf.reshape(input, [-1, input.shape[3]])
          	//console.log("Gram matrix: "+tf.transpose(input).shape+ " * "+ input.shape + " = "+gram.shape) 
          	return tf.matMul(input, input, true).div(n);
	  });
      };

     async  function displayResult(weights){
			//await tf.nextFrame();
			tf.tidy( ()  => {
				// eliminate negativ values
				let lw = weights.clone();
				lw = tf.sqrt(weights.mul(lw));
				// denorm norm from VGG19
				//weights = tf.add(weights, normV)
				//  scale too 255 range and to type in

				lw  = lw.div(tf.max(lw)).mul(255).asType("int32");
				lw = lw.concat(tf.expandDims(tf.zeros([IMAGEDIM,IMAGEDIM,1])), axis=-1);
				lw = lw.as3D(IMAGEDIM,IMAGEDIM,4)

				lw.data().then((img)  => {
					img = Uint8ClampedArray.from(img);
					let imgData = canvasContext.createImageData(IMAGEDIM, IMAGEDIM);
					for (let i = 0; i < imgData.data.length; i += 4) {
  						imgData.data[i + 0] = img[i];
  						imgData.data[i + 1] = img[i+1];
  						imgData.data[i + 2] = img[i+2];
  						imgData.data[i + 3] = 255;
					}
					createImageBitmap(imgData).then((img2)  => {
						canvasContext.drawImage(img2, 0,0, canvas.width, canvas.height);
						img2.close();
					});
				});
			});

	}

	async function trainBatch(opt, weights, features){
		opt.minimize(()  => {
				const weightsFeatures = model.predict(weights);
				const cLoss = calcContentLoss(weightsFeatures, features);
				const sLoss = calcStyleLoss(weightsFeatures, features);
				const result = cLoss.add(sLoss.mul(STYLEWEIGHT));
				return result;
			}, false, [weights]);

		await tf.nextFrame();
	}

	async function doTrain(opt, weights, features){
		let btn = document.getElementById("startTransforming");
		btn.classList.add("disabled");
		for(let step = 0; step < MAXSTEPS; step++){
			//console.log(tf.memory().numTensors)
			btn.text = (step+1)+"/"+MAXSTEPS;
			if(step % 5 == 0)
				await displayResult(weights);
			await trainBatch(opt, weights, features);

		};
	}

      async function initTransform(){

     	IMAGEDIM = parseInt($("#imagedim").val());
      	MAXSTEPS =  parseInt($("#steps").val());
     	LEARNRATE =  parseFloat($("#tfrate").val());
      	STYLEWEIGHT =  parseFloat($("#styleweight").val());


        console.log("Starting with transforming "+LEARNRATE+MAXSTEPS);
	console.log("Loading images...");

        const contentFeatures =  model.predict(getImage("content"))[4];
        const styleFeatures =  model.predict(getImage("style")).slice(0,4).map(gramMatrix);
        //const minVals = tf.mul(normV, tf.scalar(-1))
        //const maxVals = tf.sub(tf.scalar(255), normV)
        const opt = tf.train.adam(LEARNRATE, 0.99, 0.99, 1e-1);

        console.log("Content: "+contentFeatures.shape);

        for(s in styleFeatures){
          console.log("Style: " + styleFeatures[s].shape+"\n");
        }

        let weights = getImage("content"); // result of NST
	weights = tf.variable(weights);
	await doTrain(opt, weights, [contentFeatures, styleFeatures]);
	await displayResult(weights);
      }


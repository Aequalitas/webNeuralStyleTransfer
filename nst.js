// Leon Gaty`s neural style transfer with Tensorflow.js
      let model = null;
      const IMAGEDIM = 128;
      const MAXSTEPS = 3;
      // normalizing vector for the VGG19 model
      const normV = tf.tensor1d([100.939, 116.779, 123.68])


      console.log("Loading Model...")
      // first 5 are the style layers and the last is the content layer
      model = tf.loadModel('models/VGG19/model.json').then(loadedModel);

      function loadedModel(_model) 
      {

   
        model = _model;
        console.log("Model loaded! Current Backend: "+tf.getBackend());
	$("#startTransforming").click(initTransform)

      }
      
      function getImage(type){
       
        image = tf.fromPixels(document.getElementById(type+"Image"), 3).asType("float32")
        image = tf.image.resizeNearestNeighbor(image, [IMAGEDIM, IMAGEDIM])
        image = tf.expandDims(image)//.add(normV)
        //console.log(type+" image: "+image.shape)
        return image;
      }
      
      const calcLoss = (weights, features, step) =>
      {
          let weightsFeatures = model.predict(weights)
          
          // content loss
          let weightsContent = weightsFeatures[4]
          let contentLoss = tf.losses.meanSquaredError(weightsContent, features[0])
          //console.log("content loss: ("+weightsContent.shape+ " - "+ features[0].shape+") => "+contentLoss)

          // style loss
          let weightsGram = gramMatrix(weightsFeatures[0])
          let styleLoss = tf.losses.meanSquaredError(features[1][0], weightsGram)
          //console.log("style loss: ("+weightsGram.shape+ " - "+ features[1][0].shape+" ) => "+styleLoss)

          for(let s=1; s < 4; s++){
            weightsGram = gramMatrix(weightsFeatures[s])
            //console.log("style loss: MeanSquaredError -> "+styleLoss.shape + " + ("+weightsGram.shape+ " - "+ features[1][s].shape+" )")
            styleLoss = tf.losses.meanSquaredError(weightsGram, features[1][s]).add(styleLoss).div(2);
          }

	  if(step % 10 == 0)
	  	console.log("Step: "+step+" || CONTENTLOSS :" + contentLoss.dataSync()+"||  STYLELOSS : " + styleLoss.dataSync())

	return contentLoss.add(styleLoss.mul(0.1))
      }

      function gramMatrix(input){
          let n = tf.scalar(input.shape[0]).asType("float32")
          input = tf.reshape(input, [-1, input.shape[3]])
          //console.log("Gram matrix: "+tf.transpose(input).shape+ " * "+ input.shape + " = "+gram.shape) 
          return tf.matMul(input, input, true).div(n);
      }

      function displayResult(weights){

	}

      function initTransform(){
        console.log("Starting with transforming...")
       
        console.log("Loading images...")     
        const contentFeatures =  model.predict(getImage("content"))[4]
        const styleFeatures =  model.predict(getImage("style")).slice(0,4).map(gramMatrix)
        const minVals = tf.mul(normV, tf.scalar(-1))
        const maxVals = tf.sub(tf.scalar(255), normV)
        const opt = tf.train.adam(3, 0.99, 0.99, 1e-1);

        console.log("Content: "+contentFeatures.shape)
 
        for(s in styleFeatures){
          console.log("Style: " + styleFeatures[s].shape+"\n");
        }
        
        let weights = getImage("content") // result of NST
	weights = tf.variable(weights)
	//tf.tidy(() => {return calcLoss(weights, [contentFeatures, styleFeatures])}).print(true)
	for(step = 0; step < 300; step++){
          opt.minimize(() => { return tf.tidy(() => {
					return calcLoss(weights, [contentFeatures, styleFeatures], step)
					})},
					[weights]
				);
	  //tf.tidy(() => displayResult(weights))
        }


	// eliminate negativ values
	weights = tf.sqrt(weights.mul(weights))
	// denorm norm from VGG19
	weights = tf.add(weights, normV)
	//  scale too 255 range and to type int
	weights = weights.div(tf.max(weights)).mul(255).asType("int32")
	tf.toPixels(weights.as3D(IMAGEDIM,IMAGEDIM,3), document.getElementById("canvas"))

      }


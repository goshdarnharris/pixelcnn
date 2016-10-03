##Pixel CNN
Torch implementation of DeepMind's PixelCNN. Based on [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328) and [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759).

This implementation currently uses images derived from the [TGIF](https://arxiv.org/abs/1604.02748) dataset.

##Limitations
This network does not currently implement a conditional bias.

##Dependencies
- [display](https://github.com/szym/display)
- [nngraph](https://github.com/torch/nngraph)
- [csvigo](https://github.com/clementfarabet/lua---csv)

```
$ luarocks install nngraph
$ luarocks install csvigo
$ luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec
```

##Todo
- conditional bias
- imagenet loader
- sampling script
- clean up training script 
- results

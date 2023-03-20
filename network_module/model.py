import torch 

class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, bias=True, instance_norm=True):
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride, bias=bias)
        self.norm = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = torch.nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class ConvTransposed(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=2, output_padding=1, with_nonlinearity=True, bias=True, instance_norm=True):
        super(ConvTransposed, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride, output_padding=output_padding, bias=bias)
        self.norm = torch.nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = torch.nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, instance_norm):
        super(Block, self).__init__()

        self.conv_block_1 = Conv(in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, bias=False, instance_norm=instance_norm)
        self.conv_block_2 = Conv(out_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, bias=False, instance_norm=instance_norm)
        
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x 

class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, simple, instance_norm):
        super(ResBlock, self).__init__()

        self.simple = simple

        self.conv_block_1 = Conv(in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, bias=False, instance_norm=instance_norm)
        self.conv_block_2 = Conv(out_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=False, bias=False, instance_norm=instance_norm)

        self.id_conv = Conv(in_channels, out_channels, padding=0, kernel_size=1, stride=1, with_nonlinearity=False, bias=False, instance_norm=instance_norm)

        self.relu = torch.nn.ReLU()

        self.conv_block_3 = Conv(out_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=False, bias=False, instance_norm=instance_norm)
        self.conv_block_4 = Conv(out_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=False, bias=False, instance_norm=instance_norm)

    def forward(self, x):
        x1 = self.conv_block_1(x)
        x1 = self.conv_block_2(x1)
        if self.simple==True:
            x2 = x 
        else:
            x2 = self.id_conv(x)
        x = self.relu(x1+x2)

        x2 = x 
        x1 = self.conv_block_3(x)
        x1 = self.conv_block_4(x1)
        x = self.relu(x1+x2)

        return x 

class Down(torch.nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.mp = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.mp(x)
        return x 

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, instance_norm):
        super(Up, self).__init__()
        self.conv = ConvTransposed(in_channels, out_channels, padding=1, kernel_size=3, stride=2, output_padding=1, with_nonlinearity=False, bias=False, instance_norm=instance_norm)
                # for reproducibility

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, instance_norm):
        super(UNet, self).__init__()

        self.in_conv = Conv(in_channels, 64, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, bias=False, instance_norm=instance_norm)
        
        self.down_block_1 = Block(64, 64, instance_norm=instance_norm)
        self.down_block_2 = Block(64, 128, instance_norm=instance_norm)
        self.down_block_3 = Block(128, 256, instance_norm=instance_norm)
        self.down_block_4 = Block(256, 512, instance_norm=instance_norm)

        self.block_5 = Block(512, 1024, instance_norm=instance_norm)

        self.up_block_4 = Block(1024, 512, instance_norm=instance_norm)
        self.up_block_3 = Block(512, 256, instance_norm=instance_norm)
        self.up_block_2 = Block(256, 128, instance_norm=instance_norm)
        self.up_block_1 = Block(128, 64, instance_norm=instance_norm)

        self.down = Down()

        self.up4 = Up(1024, 512, instance_norm=instance_norm)
        self.up3 = Up(512, 256, instance_norm=instance_norm)
        self.up2 = Up(256, 128, instance_norm=instance_norm)
        self.up1 = Up(128, 64, instance_norm=instance_norm)

        self.out_conv = torch.nn.Conv2d(64, out_channels, padding=1, kernel_size=3, stride=1)

    def forward(self, x):
        # in: batch x 3 x 16h x 16w
        # out: batch x 64 x 16h x 16w
        x = self.in_conv(x)         

        # out: batch x 64 x 16h x 16w
        x1 = self.down_block_1(x)

        # out: batch x 64 x 8h x 8w
        x2 = self.down(x1)
        # out: batch x 128 x 8h x 8w
        x2 = self.down_block_2(x2)

        # out: batch x 128 x 4h x 4w
        x3 = self.down(x2)
        # out: batch x 256 x 4h x 4w
        x3 = self.down_block_3(x3)

        # out: batch x 256 x 2h x 2w
        x4 = self.down(x3)
        # out: batch x 512 x 2h x 2w
        x4 = self.down_block_4(x4)

        # out: batch x 512 x h x w
        x5 = self.down(x4)
        # out: batch x 1024 x h x w
        x5 = self.block_5(x5)

        # out: batch x 512 x 2h x 2w
        x = self.up4(x5)
        # out: batch x 1024 x 2h x 2w
        x = torch.cat((x4, x), dim=1)
        # out: batch x 512 x 2h x 2w
        x = self.up_block_4(x)

        # out: batch x 256 x 4h x 4w
        x = self.up3(x)
        # out: batch x 512 x 4h x 4w
        x = torch.cat((x3, x), dim=1)
        # out: batch x 256 x 4h x 4w
        x = self.up_block_3(x)

        # out: batch x 128 x 8h x 8w
        x = self.up2(x)
        # out: batch x 256 x 8h x 8w
        x = torch.cat((x2, x), dim=1)
        # out: batch x 128 x 8h x 8w
        x = self.up_block_2(x)

        # out: batch x 64 x 16h x 16w
        x = self.up1(x)
        # out: batch x 128 x 16h x 16w
        x = torch.cat((x1, x), dim=1)
        # out: batch x 64 x 16h x 16w
        x = self.up_block_1(x)

        # out: batch x 2 x 16h x 16w
        x = self.out_conv(x)

        return x

class ResNet_UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, instance_norm=True):
        super(ResNet_UNet, self).__init__()

        self.in_conv = Conv(in_channels, 64, padding=3, kernel_size=7, stride=1, with_nonlinearity=True, bias=False, instance_norm=instance_norm)
   
        self.down_block_1 = ResBlock(64, 64, simple=True, instance_norm=instance_norm)
        self.down_block_2 = ResBlock(64, 128, simple=False, instance_norm=instance_norm)
        self.down_block_3 = ResBlock(128, 256, simple=False, instance_norm=instance_norm)
        self.down_block_4 = ResBlock(256, 512, simple=False, instance_norm=instance_norm)

        self.block_5 = ResBlock(512, 1024, simple=False, instance_norm=instance_norm)

        self.up_block_4 = ResBlock(1024, 512, simple=False, instance_norm=instance_norm)
        self.up_block_3 = ResBlock(512, 256, simple=False, instance_norm=instance_norm)
        self.up_block_2 = ResBlock(256, 128, simple=False, instance_norm=instance_norm)
        self.up_block_1 = ResBlock(128, 64, simple=False, instance_norm=instance_norm)

        self.down = Down()

        self.up4 = Up(1024, 512, instance_norm=instance_norm)
        self.up3 = Up(512, 256, instance_norm=instance_norm)
        self.up2 = Up(256, 128, instance_norm=instance_norm)
        self.up1 = Up(128, 64, instance_norm=instance_norm)

        self.out_conv = torch.nn.Conv2d(64, out_channels, padding=1, kernel_size=3, stride=1)
        
    def set_weights_from_pretrained_model(self,pretrained_model):
        # initial conv 
        self.in_conv.conv.weight.data = pretrained_model.conv1.weight.data
        self.in_conv.batch_norm.weight.data = pretrained_model.bn1.weight.data
        self.in_conv.batch_norm.bias.data = pretrained_model.bn1.bias.data

        # down 1 
        # 0
        layer_data = pretrained_model.layer1
        self.down_block_1.conv_block_1.conv.weight.data = layer_data[0].conv1.weight.data
        self.down_block_1.conv_block_1.norm.weight.data = layer_data[0].bn1.weight.data
        self.down_block_1.conv_block_1.norm.bias.data = layer_data[0].bn1.bias.data

        self.down_block_1.conv_block_2.conv.weight.data = layer_data[0].conv2.weight.data
        self.down_block_1.conv_block_2.norm.weight.data = layer_data[0].bn2.weight.data
        self.down_block_1.conv_block_2.norm.bias.data = layer_data[0].bn2.bias.data
        # 1
        self.down_block_1.conv_block_3.conv.weight.data = layer_data[1].conv1.weight.data
        self.down_block_1.conv_block_3.norm.weight.data = layer_data[1].bn1.weight.data
        self.down_block_1.conv_block_3.norm.bias.data = layer_data[1].bn1.bias.data

        self.down_block_1.conv_block_4.conv.weight.data = layer_data[1].conv2.weight.data
        self.down_block_1.conv_block_4.norm.weight.data = layer_data[1].bn2.weight.data
        self.down_block_1.conv_block_4.norm.bias.data = layer_data[1].bn2.bias.data

        # down 2
        # 0
        layer_data = pretrained_model.layer2
        self.down_block_2.conv_block_1.conv.weight.data = layer_data[0].conv1.weight.data
        self.down_block_2.conv_block_1.norm.weight.data = layer_data[0].bn1.weight.data
        self.down_block_2.conv_block_1.norm.bias.data = layer_data[0].bn1.bias.data

        self.down_block_2.conv_block_2.conv.weight.data = layer_data[0].conv2.weight.data
        self.down_block_2.conv_block_2.norm.weight.data = layer_data[0].bn2.weight.data
        self.down_block_2.conv_block_2.norm.bias.data = layer_data[0].bn2.bias.data

        self.down_block_2.in_conv.conv.weight.data = layer_data[0].downsample[0].weight.data
        self.down_block_2.in_conv.norm.weight.data = layer_data[0].downsample[1].weight.data
        self.down_block_2.in_conv.norm.bias.data = layer_data[0].downsample[1].bias.data
        # 1
        self.down_block_2.conv_block_3.conv.weight.data = layer_data[1].conv1.weight.data
        self.down_block_2.conv_block_3.norm.weight.data = layer_data[1].bn1.weight.data
        self.down_block_2.conv_block_3.norm.bias.data = layer_data[1].bn1.bias.data

        self.down_block_2.conv_block_4.conv.weight.data = layer_data[1].conv2.weight.data
        self.down_block_2.conv_block_4.norm.weight.data = layer_data[1].bn2.weight.data
        self.down_block_2.conv_block_4.norm.bias.data = layer_data[1].bn2.bias.data


        # down 3
        # 0
        layer_data = pretrained_model.layer3
        self.down_block_3.conv_block_1.conv.weight.data = layer_data[0].conv1.weight.data
        self.down_block_3.conv_block_1.norm.weight.data = layer_data[0].bn1.weight.data
        self.down_block_3.conv_block_1.norm.bias.data = layer_data[0].bn1.bias.data

        self.down_block_3.conv_block_2.conv.weight.data = layer_data[0].conv2.weight.data
        self.down_block_3.conv_block_2.norm.weight.data = layer_data[0].bn2.weight.data
        self.down_block_3.conv_block_2.norm.bias.data = layer_data[0].bn2.bias.data

        self.down_block_3.in_conv.conv.weight.data = layer_data[0].downsample[0].weight.data
        self.down_block_3.in_conv.norm.weight.data = layer_data[0].downsample[1].weight.data
        self.down_block_3.in_conv.norm.bias.data = layer_data[0].downsample[1].bias.data
        # 1
        self.down_block_3.conv_block_3.conv.weight.data = layer_data[1].conv1.weight.data
        self.down_block_3.conv_block_3.norm.weight.data = layer_data[1].bn1.weight.data
        self.down_block_3.conv_block_3.norm.bias.data = layer_data[1].bn1.bias.data

        self.down_block_3.conv_block_4.conv.weight.data = layer_data[1].conv2.weight.data
        self.down_block_3.conv_block_4.norm.weight.data = layer_data[1].bn2.weight.data
        self.down_block_3.conv_block_4.norm.bias.data = layer_data[1].bn2.bias.data

        # down 4
        # 0
        layer_data = pretrained_model.layer4
        self.down_block_4.conv_block_1.conv.weight.data = layer_data[0].conv1.weight.data
        self.down_block_4.conv_block_1.norm.weight.data = layer_data[0].bn1.weight.data
        self.down_block_4.conv_block_1.norm.bias.data = layer_data[0].bn1.bias.data

        self.down_block_4.conv_block_2.conv.weight.data = layer_data[0].conv2.weight.data
        self.down_block_4.conv_block_2.norm.weight.data = layer_data[0].bn2.weight.data
        self.down_block_4.conv_block_2.norm.bias.data = layer_data[0].bn2.bias.data

        self.down_block_4.in_conv.conv.weight.data = layer_data[0].downsample[0].weight.data
        self.down_block_4.in_conv.norm.weight.data = layer_data[0].downsample[1].weight.data
        self.down_block_4.in_conv.norm.bias.data = layer_data[0].downsample[1].bias.data
        # 1
        self.down_block_4.conv_block_3.conv.weight.data = layer_data[1].conv1.weight.data
        self.down_block_4.conv_block_3.norm.weight.data = layer_data[1].bn1.weight.data
        self.down_block_4.conv_block_3.norm.bias.data = layer_data[1].bn1.bias.data

        self.down_block_4.conv_block_4.conv.weight.data = layer_data[1].conv2.weight.data
        self.down_block_4.conv_block_4.norm.weight.data = layer_data[1].bn2.weight.data
        self.down_block_4.conv_block_4.norm.bias.data = layer_data[1].bn2.bias.data

    def forward(self, x):
        # in: batch x 3 x 16h x 16w
        # out: batch x 64 x 16h x 16w
        x = self.in_conv(x)         

        # out: batch x 64 x 16h x 16w
        x1 = self.down_block_1(x)

        # out: batch x 64 x 8h x 8w
        x2 = self.down(x1)
        # out: batch x 128 x 8h x 8w
        x2 = self.down_block_2(x2)

        # out: batch x 128 x 4h x 4w
        x3 = self.down(x2)
        # out: batch x 256 x 4h x 4w
        x3 = self.down_block_3(x3)

        # out: batch x 256 x 2h x 2w
        x4 = self.down(x3)
        # out: batch x 512 x 2h x 2w
        x4 = self.down_block_4(x4)

        # out: batch x 512 x h x w
        x5 = self.down(x4)
        # out: batch x 1024 x h x w
        x5 = self.block_5(x5)

        # out: batch x 512 x 2h x 2w
        x = self.up4(x5)
        # out: batch x 1024 x 2h x 2w
        x = torch.cat((x4, x), dim=1)
        # out: batch x 512 x 2h x 2w
        x = self.up_block_4(x)

        # out: batch x 256 x 4h x 4w
        x = self.up3(x)
        # out: batch x 512 x 4h x 4w
        x = torch.cat((x3, x), dim=1)
        # out: batch x 256 x 4h x 4w
        x = self.up_block_3(x)

        # out: batch x 128 x 8h x 8w
        x = self.up2(x)
        # out: batch x 256 x 8h x 8w
        x = torch.cat((x2, x), dim=1)
        # out: batch x 128 x 8h x 8w
        x = self.up_block_2(x)

        # out: batch x 64 x 16h x 16w
        x = self.up1(x)
        # out: batch x 128 x 16h x 16w
        x = torch.cat((x1, x), dim=1)
        # out: batch x 64 x 16h x 16w
        x = self.up_block_1(x)

        # out: batch x 2 x 16h x 16w
        x = self.out_conv(x)

        return x

const std = @import("std");
const logger = @import("logger.zig");
const Layers = @import("layers.zig");

pub fn CNN(
  F: type,
  height: comptime_int, width: comptime_int,
  loss_gen: fn(LEN: comptime_int, F: type) type,
  layers: anytype,
) type {
  @setEvalBranchQuota(1000_000);
  const mergedTrainer = Layers.mergeAny(layers)(F, height, width, true);
  std.debug.assert(mergedTrainer.height == 1);

  const mergedTester = Layers.mergeAny(layers)(F, height, width, false);
  std.debug.assert(mergedTester.height == mergedTrainer.height);
  std.debug.assert(mergedTester.width == mergedTrainer.width);

  const Retval = struct {
    pub const OutputWidth = mergedTrainer.width;
    const LossFn = loss_gen(OutputWidth, F);

    pub const Gradient = mergedTrainer.layer.Gradient;

    pub const Trainer = struct {
      layer: mergedTrainer,


      pub fn toTester(self: *const @This()) Tester {
        return .{.layer = mergedTester.layer.fromLayer(self.layer.asBytes())};
      }

      // Sets the value in cache
      pub inline fn forward(self: *@This(), input: *[height][width]F, output: *[1][OutputWidth]F) void {
        @import("read_minst.zig").printImage(input);
        self.layer.forward(input, output);
      }

      pub fn backward(self: *@This(), input: *[height][width]F, output: *[1][OutputWidth]F, target: u8, gradient: *Gradient) void {
        var d_buf: [OutputWidth]F = undefined;
        LossFn.backward(&output[0], target, &d_buf);
        logger.log(&@src(), "predictions: {d}\n", .{&output[0]});
        logger.log(&@src(), "--- dLoss ({d}) ---\n\t{any}\n", .{target, &d_buf});

        self.layer.backward(input, output, undefined, &d_buf, gradient, false);
      }

      pub fn reset(self: *@This(), rng: std.Random) void {
        self.layer.reset(rng);
      }

      pub const Options = struct {
        verbose: bool = true,
        batch_size: u32,
        learning_rate: F,
      };
      pub fn train(self: *@This(), iterator_readonly: anytype, options: Options) void {
        var iterator = iterator_readonly;
        var input_buf: [height][width]F = undefined;
        var output_buf: [1][OutputWidth]F = undefined;
        var gradients: Gradient = undefined;

        // inline for (0..@This().Layers.len) |i| {
        //   logger.log(&@src(), "Layer {d}\n", .{i});
        //   logger.log(&@src(), "{any}\n", .{@field(self.layers, std.fmt.comptimePrint("{d}", .{i}))});
        // }

        var step: usize = 0;
        while (true) {
          var n: usize = 0;
          var gross_loss: F = 0;

          step += 1;
          gradients.reset();

          for (0..options.batch_size) |_| {
            const next = iterator.next() orelse break;

            inline for (next.image, 0..) |row, i| {
              inline for (row, 0..) |val, j| {
                input_buf[i][j] = @as(F, @floatFromInt(val));
              }
            }

            n += 1;
            self.forward(&input_buf, &output_buf);
            self.backward(&input_buf, &output_buf, next.label, &gradients);

            const loss = LossFn.forward(output_buf, next.label);
            gross_loss += loss;
            if (options.verbose) {
              logger.writer.print("{d:4}-{d:2} Loss({d}) = {d:.3}\n", .{step, n, next.label, loss*100}) catch {};
              // logger.log(&@src(), "Gradients {any}\n", .{gradients});
            }
          }

          if (!options.verbose) {
            logger.writer.print("Step: {d:4}-(1-{d:2}) Loss: {d:.3}\n", .{step, n, gross_loss*100}) catch {};
          }

          self.applyGradients(&gradients, options.learning_rate / @as(F, @floatFromInt(options.batch_size)));

          logger.buffered.flush() catch {};
          if (!iterator.hasNext()) break;
        }
      }
    };

    pub const Tester = struct {
      layer: mergedTester,

      pub fn toTrainer(self: *const @This()) Trainer {
        return .{.layer = mergedTrainer.layer.fromLayer(self.layer.asBytes())};
      }

      pub fn forward(self: *const @This(), input: *[height][width]F, output: *[1][OutputWidth]F) void {
        self.layer.forward(input, output);
      }

      pub fn @"test"(self: *const @This(), iterator_ro: anytype) F {
        defer logger.buffered.flush() catch {};
        var retval: usize = 0;
        var n: usize = 0;
        var iterator = iterator_ro;
        while (iterator.next()) |next| {
          n += 1;
          const predictions = self.forward(next.image.*);
          var guess: usize = 0;
          inline for (0..predictions.len) |i| {
            if (predictions[guess] < predictions[i]) guess = i;
          }
          if (guess == next.label) retval += 1;

          logger.writer.print("{d:5} Prediction({d}) = {d}\n", .{n, next.label, guess}) catch {};
        }

        return @as(F, @floatFromInt(retval)) / @as(F, @floatFromInt(n));
      }
    };

    test {
      std.debug.print("\n----------- TRAINING LAYERS -----------\n", .{});
      inline for (Trainer.Layers) |l| std.debug.print("{any}\n", .{l});
      std.debug.print("\n----------- TESTING LAYERS -----------\n", .{});
      inline for (Tester.Layers) |l| std.debug.print("{any}\n", .{l});
    }
  };

  std.debug.assert(Retval.Trainer.Layers[Retval.Trainer.Layers.len - 1].output_height == 1);
  return Retval;
}

test "CNN Deepnet" {
  const Layer = @import("layers.zig");
  const Activation = @import("functions_activate.zig");
  const Loss = @import("functions_loss.zig");

  const Iterator = struct {
    rng: std.Random,
    remaining: u32,
    val: [1][3]u8 = undefined,
    repetitions: u32,
    remaining_repetitions: u32 = 0,

    fn next(self: *@This()) ?struct {
      image: *[1][3]u8,
      label: u8,
    } {
      if (self.remaining_repetitions > 0) {
        self.remaining_repetitions -= 1;
      } else if (self.remaining > 0) {
        self.remaining -= 1;
        self.remaining_repetitions = self.repetitions;
        self.val[0][0] = self.rng.intRangeLessThan(u8, 0, 2);
        self.val[0][1] = self.rng.intRangeLessThan(u8, 0, 2);
        self.val[0][2] = self.rng.intRangeLessThan(u8, 0, 2);
      } else {
        return null;
      }

      return .{
        .image = &self.val,
        .label = (self.val[0][0] ^ self.val[0][1]) + self.val[0][2],
      };
    }

    pub fn hasNext(self: *const @This()) bool {
      return self.remaining_repetitions > 0 or self.remaining > 0;
    }
  };

  // xor test
  const cnn = CNN(f64, 1, 3, Loss.MeanSquaredError, [_]Layer.LayerType{
    Layer.getDense(9, Activation.getPReLU(0.1)),
    Layer.getDense(9, Activation.getPReLU(0.1)),
    Layer.getDense(6, Activation.getPReLU(0.1)),
    Layer.getDense(3, Activation.NormalizeSquared),
  });

  var trainer = cnn.Trainer{.layers = undefined};

  var rng = std.Random.DefaultPrng.init(0);
  trainer.reset(rng.random());

  for (0..16) |i| {
    trainer.train(Iterator{.rng = rng.random(), .remaining = 1024, .repetitions = 0}, .{
      .verbose = true,
      .batch_size = @intCast(16),
      .learning_rate = @as(f64, 1.4) / @as(f64, @floatFromInt(1 + i)),
    });
  }

  var tester = trainer.toTester();
  const accuracy = tester.@"test"(Iterator{.rng = rng.random(), .remaining = 512, .repetitions = 0});
  std.debug.print("\n>> Final Accuracy: {d:.3}%\n", .{accuracy*100});
}


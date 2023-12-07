open Ctypes
open Utils

module Bindings (F : FOREIGN) = struct
  open F

  (* Registration for the entire group *)
  let register_passes = foreign "mlirRegisterConversionPasses" (void @-> returning void)

  module ArithToLLVM =
    MakeConversion
      (struct
        let s = "ArithToLLVMConversionPass"
      end)
      (F)

  module AMDGPUToROCDL =
    MakeConversion
      (struct
        let s = "ConvertAMDGPUToROCDL"
      end)
      (F)

  module AffineForToGPU =
    MakeConversion
      (struct
        let s = "ConvertAffineForToGPU"
      end)
      (F)

  module AffineToStandard =
    MakeConversion
      (struct
        let s = "ConvertAffineToStandard"
      end)
      (F)

  module ArithToSPIRV =
    MakeConversion
      (struct
        let s = "ConvertArithToSPIRV"
      end)
      (F)

  module ArmNeon2dToIntr =
    MakeConversion
      (struct
        let s = "ConvertArmNeon2dToIntr"
      end)
      (F)

  module AsyncToLLVM =
    MakeConversion
      (struct
        let s = "ConvertAsyncToLLVM"
      end)
      (F)

  module BufferizationToMemRef =
    MakeConversion
      (struct
        let s = "ConvertBufferizationToMemRef"
      end)
      (F)

  module ComplexToLLVM =
    MakeConversion
      (struct
        let s = "ConvertComplexToLLVM"
      end)
      (F)

  module ComplexToLibm =
    MakeConversion
      (struct
        let s = "ConvertComplexToLibm"
      end)
      (F)

  module ComplexToStandard =
    MakeConversion
      (struct
        let s = "ConvertComplexToStandard"
      end)
      (F)

  module ControlFlowToLLVM =
    MakeConversion
      (struct
        let s = "ConvertControlFlowToLLVM"
      end)
      (F)

  module ControlFlowToSPIRV =
    MakeConversion
      (struct
        let s = "ConvertControlFlowToSPIRV"
      end)
      (F)

  module FuncToLLVM =
    MakeConversion
      (struct
        let s = "ConvertFuncToLLVM"
      end)
      (F)

  module FuncToSPIRV =
    MakeConversion
      (struct
        let s = "ConvertFuncToSPIRV"
      end)
      (F)

  module GPUToSPIRV =
    MakeConversion
      (struct
        let s = "ConvertGPUToSPIRV"
      end)
      (F)

  module GpuLaunchFuncToVulkanLaunchFunc =
    MakeConversion
      (struct
        let s = "ConvertGpuLaunchFuncToVulkanLaunchFunc"
      end)
      (F)

  module GpuOpsToNVVMOps =
    MakeConversion
      (struct
        let s = "ConvertGpuOpsToNVVMOps"
      end)
      (F)

  module GpuOpsToROCDLOps =
    MakeConversion
      (struct
        let s = "ConvertGpuOpsToROCDLOps"
      end)
      (F)

  module IndexToLLVMPass =
    MakeConversion
      (struct
        let s = "ConvertIndexToLLVMPass"
      end)
      (F)

  module LinalgToLLVM =
    MakeConversion
      (struct
        let s = "ConvertLinalgToLLVM"
      end)
      (F)

  module LinalgToStandard =
    MakeConversion
      (struct
        let s = "ConvertLinalgToStandard"
      end)
      (F)

  module MathToFuncs =
    MakeConversion
      (struct
        let s = "ConvertMathToFuncs"
      end)
      (F)

  module MathToLLVM =
    MakeConversion
      (struct
        let s = "ConvertMathToLLVM"
      end)
      (F)

  module MathToLibm =
    MakeConversion
      (struct
        let s = "ConvertMathToLibm"
      end)
      (F)

  module MathToSPIRV =
    MakeConversion
      (struct
        let s = "ConvertMathToSPIRV"
      end)
      (F)

  module MemRefToSPIRV =
    MakeConversion
      (struct
        let s = "ConvertMemRefToSPIRV"
      end)
      (F)

  module NVGPUToNVVM =
    MakeConversion
      (struct
        let s = "ConvertNVGPUToNVVM"
      end)
      (F)

  module OpenACCToLLVM =
    MakeConversion
      (struct
        let s = "ConvertOpenACCToLLVM"
      end)
      (F)

  module OpenACCToSCF =
    MakeConversion
      (struct
        let s = "ConvertOpenACCToSCF"
      end)
      (F)

  module OpenMPToLLVM =
    MakeConversion
      (struct
        let s = "ConvertOpenMPToLLVM"
      end)
      (F)

  module PDLToPDLInterp =
    MakeConversion
      (struct
        let s = "ConvertPDLToPDLInterp"
      end)
      (F)

  module ParallelLoopToGpu =
    MakeConversion
      (struct
        let s = "ConvertParallelLoopToGpu"
      end)
      (F)

  module SCFToOpenMP =
    MakeConversion
      (struct
        let s = "ConvertSCFToOpenMP"
      end)
      (F)

  module SPIRVToLLVM =
    MakeConversion
      (struct
        let s = "ConvertSPIRVToLLVM"
      end)
      (F)

  module ShapeConstraints =
    MakeConversion
      (struct
        let s = "ConvertShapeConstraints"
      end)
      (F)

  module ShapeToStandard =
    MakeConversion
      (struct
        let s = "ConvertShapeToStandard"
      end)
      (F)

  module TensorToLinalg =
    MakeConversion
      (struct
        let s = "ConvertTensorToLinalg"
      end)
      (F)

  module TensorToSPIRV =
    MakeConversion
      (struct
        let s = "ConvertTensorToSPIRV"
      end)
      (F)

  module VectorToGPU =
    MakeConversion
      (struct
        let s = "ConvertVectorToGPU"
      end)
      (F)

  module VectorToLLVM =
    MakeConversion
      (struct
        let s = "ConvertVectorToLLVM"
      end)
      (F)

  module VectorToSCF =
    MakeConversion
      (struct
        let s = "ConvertVectorToSCF"
      end)
      (F)

  module VectorToSPIRV =
    MakeConversion
      (struct
        let s = "ConvertVectorToSPIRV"
      end)
      (F)

  module VulkanLaunchFuncToVulkanCalls =
    MakeConversion
      (struct
        let s = "ConvertVulkanLaunchFuncToVulkanCalls"
      end)
      (F)

  module GpuToLLVMConversionPass =
    MakeConversion
      (struct
        let s = "GpuToLLVMConversionPass"
      end)
      (F)

  module LowerHostCodeToLLVM =
    MakeConversion
      (struct
        let s = "LowerHostCodeToLLVM"
      end)
      (F)

  module MapMemRefStorageClass =
    MakeConversion
      (struct
        let s = "MapMemRefStorageClass"
      end)
      (F)

  module MemRefToLLVMConversionPass =
    MakeConversion
      (struct
        let s = "MemRefToLLVMConversionPass"
      end)
      (F)

  module ReconcileUnrealizedCasts =
    MakeConversion
      (struct
        let s = "ReconcileUnrealizedCasts"
      end)
      (F)

  module SCFToControlFlow =
    MakeConversion
      (struct
        let s = "SCFToControlFlow"
      end)
      (F)

  module SCFToSPIRV =
    MakeConversion
      (struct
        let s = "SCFToSPIRV"
      end)
      (F)

  module TosaToArith =
    MakeConversion
      (struct
        let s = "TosaToArith"
      end)
      (F)

  module TosaToLinalg =
    MakeConversion
      (struct
        let s = "TosaToLinalg"
      end)
      (F)

  module TosaToLinalgNamed =
    MakeConversion
      (struct
        let s = "TosaToLinalgNamed"
      end)
      (F)

  module TosaToSCF =
    MakeConversion
      (struct
        let s = "TosaToSCF"
      end)
      (F)

  module TosaToTensor =
    MakeConversion
      (struct
        let s = "TosaToTensor"
      end)
      (F)
end

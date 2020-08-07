#pragma once

#include <vector>
#include <memory>

#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <engine/Hamiltonian.hpp>
#include <data/Geometry.hpp>
#include "engine/FFT.hpp"
#include <vulkan/vulkan.h>
#include "VulkanInitializers.hpp"
#include <string.h>
#include <chrono>

#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}
#define SUPPORTED_RADIX_LEVELS 1
#define COUNT_OF(array) (sizeof(array) / sizeof(array[0]))
#define __builtin_clz __lzcnt
using FFT_real_type = scalar;
using FFT_cpx_type = std::array<scalar, 2>;
namespace VulkanCompute
{
	typedef struct {
		VkAllocationCallbacks* allocator;
		VkPhysicalDevice physicalDevice;
		VkPhysicalDeviceProperties physicalDeviceProperties;
		VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
		VkDevice device;
		VkQueue queue;
		VkCommandPool commandPool;
		VkFence fence;
		VkShaderModule shaderModules[SUPPORTED_RADIX_LEVELS];
		VkDeviceSize uboAlignment;
	} VulkanFFTContext;

	typedef struct {
		VulkanFFTContext* context;
		VkDeviceSize size;
		VkBuffer hostBuffer, deviceBuffer;
		VkDeviceMemory deviceMemory;
	} VulkanFFTTransfer;
	typedef struct {
		uint32_t sampleCount;
		uint32_t stageCount;
		uint32_t* stageRadix;
		VkDeviceSize uboSize;
		VkBuffer ubo;
		VkDeviceMemory uboDeviceMemory;
		VkDescriptorPool descriptorPool;
		VkDescriptorSetLayout* descriptorSetLayouts;
		VkDescriptorSet* descriptorSets;
		VkPipelineLayout pipelineLayout;
		VkPipeline* pipelines;
	}VulkanFFTAxis;

	typedef struct {
		VulkanFFTContext* context;
		bool inverse, resultInSwapBuffer;
		VulkanFFTAxis axes[3];
		VkDeviceSize* bufferSize;
		VkBuffer* buffer;
		VkDeviceMemory* deviceMemory;
	} VulkanFFTPlan;
	typedef struct {
		uint32_t stride[3];
		uint32_t radixStride, stageSize;
		float directionFactor;
		float angleFactor;
		float normalizationFactor;
		uint32_t stageCount;
		uint32_t WIDTH;
		uint32_t HEIGHT;
		uint32_t DEPTH;
		uint32_t npad;
	} VulkanFFTUBO;

	typedef struct {
		VulkanFFTContext* context;
		VkDescriptorPool descriptorPool;
		VkDescriptorSetLayout* descriptorSetLayouts;
		VkDescriptorSet* descriptorSets;
		VkPipelineLayout pipelineLayout;
		VkPipeline* pipelines;
		VkShaderModule computeShader;
		VkCommandBuffer commandBuffer;
	} VulkanCollection;

	typedef struct {
		VulkanFFTContext* context;
		int bufferNum;
		int* sizes;
		VkDeviceSize bufferSizes;
		VkBuffer* buffer;
		VkDeviceMemory* deviceMemory;
		VulkanCollection collection;
	} VulkanReduce;

	typedef struct {

		VulkanFFTContext* context;
		VkDeviceSize* bufferSizes;
		VkBuffer* buffer;
		VkDeviceMemory* deviceMemory;
		scalar scaling = 1.0f;
		int n_lbfgs_memory = 3; // how many previous iterations are stored in the memory
		uint32_t m_index;
		uint32_t ReduceDotConsts[4];
		struct Apply0Consts {
			uint32_t offset1;
		} apply0Consts;
		struct Apply1Consts {
			uint32_t offset1;
			scalar alpha;
		} apply1Consts;
		struct Apply2Consts {
			uint32_t offset1;
			scalar alpha;
			scalar rhopdg;
		} apply2Consts;
		struct ApplyVP1Consts {
			scalar m_temp_inv;
		} applyVP1Consts;
		struct ApplyVP2Consts {
			scalar grad_mult;
		} applyVP2Consts;
		struct SetDir1Consts {
			scalar inv_rhody2;
		} setDir1Consts;
		VulkanCollection collectionSetDir0;
		VulkanCollection collectionSetDir1;
		VulkanCollection collectionSetdadg;
		VulkanCollection* collectionApply0;
		VulkanCollection collectionApply1;
		VulkanCollection collectionApply2;
		VulkanCollection collectionApply3;
		VulkanCollection* collectionReduceDot0;
		VulkanCollection* collectionReduceDot1;
		VulkanCollection* collectionReduceDot2;
		VulkanCollection* collectionReduceDot3;
		VulkanCollection collectionReduceDotScaling;
		VulkanCollection collectionReduceDotFinish;
		VulkanCollection collectionOsoCalcGradients;
		VulkanCollection collectionReduce;
		VulkanCollection collectionScale;
		VulkanCollection collectionOsoRotate;
	} VulkanLBFGS;

	typedef struct {
		uint32_t WIDTH;
		uint32_t HEIGHT;
		uint32_t DEPTH;
		uint32_t n;
		uint32_t npad;
		uint32_t npad2;
	} VulkanDimensions;
	class ComputeApplication {
	private:
		uint32_t SIZES[3];
		int WORKGROUP_SIZE = 32; // Workgroup size in compute shader.
		uint32_t LOCAL_SIZE[3];
		uint32_t LOCAL_SIZE_FFT[3];
		void* mapReduce;
		/*
		In order to use Vulkan, you must create an instance.
		*/
		VkInstance instance;

		VkDebugReportCallbackEXT debugReportCallback;
		/*
		The physical device is some device on the system that supports usage of Vulkan.
		Often, it is simply a graphics card that supports Vulkan.
		*/
		VkPhysicalDevice physicalDevice;
		/*
		Then we have the logical device VkDevice, which basically allows
		us to interact with the physical device.
		*/
		VkDevice device;

		/*
		The pipeline specifies the pipeline that all graphics and compute commands pass though in Vulkan.

		We will be creating a simple compute pipeline in this application.
		*/


		/*
		The command buffer is used to record commands, that will be submitted to a queue.

		To allocate such command buffers, we use a command pool.
		*/
		VkCommandPool commandPool;

		VkCommandBuffer commandBufferFFT;
		VkCommandBuffer commandBufferiFFT;
		VkCommandBuffer commandBufferCollected[7];
		VkCommandBuffer commandBufferReduce;
		VkCommandBuffer commandBufferTransferSolver;
		VkCommandBuffer commandBufferTransferSolver2;
		/*

		Descriptors represent resources in shaders. They allow us to use things like
		uniform buffers, storage buffers and images in GLSL.

		A single descriptor represents a single resource, and several descriptors are organized
		into descriptor sets, which are basically just collections of descriptors.
		*/

		/*
		The mandelbrot set will be rendered to this buffer.

		The memory that backs the buffer is bufferMemory.
		*/
		VkBuffer bufferSpins;
		VkBuffer bufferGradient;
		VkBuffer bufferSpinsInit;
		VkBuffer bufferGradientOut;
		VkBuffer bufferRegions_Book;
		VkBuffer bufferRegions;
		VkBuffer kernel;
		VkBuffer uboDimensions;
		VkDeviceMemory bufferMemorySpins;
		VkDeviceMemory bufferMemoryGradient;
		VkDeviceMemory bufferMemorySpinsInit;
		VkDeviceMemory bufferMemoryGradientOut;
		VkDeviceMemory bufferMemoryRegions_Book;
		VkDeviceMemory bufferMemoryRegions;
		VkDeviceMemory bufferMemoryKernel;
		VkDeviceMemory uboMemoryDimensions;
		uint32_t bufferSizeSpins;
		uint32_t bufferSizeGradient;
		uint32_t bufferSizeRegions_Book;
		uint32_t bufferSizeRegions;
		uint32_t bufferSizeKernel;
		uint32_t uboSizeDimensions;
		std::vector<const char*> enabledLayers;

		VkPipelineCache pipelineCache;

		VulkanCollection collectionGradients_noDDI;
		VulkanCollection collectionZeroPadding;
		VulkanCollection collectionZeroPaddingRemove;
		VulkanCollection collectionConvolution;
		VulkanCollection collectionFillZero;
		VulkanCollection collectionReadSpins;
		VulkanCollection collectionWriteGradient;
		VulkanCollection collectionWriteSpins;
		VulkanCollection collectionC2R_decomp;
		VulkanCollection collectionR2C_decomp;
		VulkanCollection collectionReduceDotEnergy;
		// DDI FFT
		VulkanFFTContext context = {};
		VulkanFFTPlan vulkanFFTPlan = { &context };
		VulkanFFTPlan vulkaniFFTPlan = { &context };
		VulkanFFTTransfer vulkanFFTTransferGradient;
		VulkanFFTTransfer vulkanFFTTransferSpins;
		VulkanFFTTransfer vulkanFFTTransferReduce;
		VulkanFFTTransfer vulkanFFTTransferReduceDot;

		VulkanReduce vulkanReduce{ &context };
		VulkanLBFGS vulkanLBFGS{ &context };
		/*
		In order to execute commands on a device(GPU), the commands must be submitted
		to a queue. The commands are stored in a command buffer, and this command buffer
		is given to the queue.

		There will be different kinds of queues on the device. Not all queues support
		graphics operations, for instance. For this application, we at least want a queue
		that supports compute operations.
		*/
		VkQueue queue; // a queue supporting compute operations.
		VkSubmitInfo submitInfoCollected = {};
		/*
		Groups of queues that have the same capabilities(for instance, they all supports graphics and computer operations),
		are grouped into queue families.

		When submitting a command buffer, you must specify to which queue in the family you are submitting to.
		This variable keeps track of the index of that queue in its family.
		*/
		uint32_t queueFamilyIndex;
		struct UboFFTDynamic {
			VulkanFFTUBO* model = nullptr;
		} uboFFTDynamic;

	public:
		VulkanFFTTransfer vulkanFFTTransferGrad;
		VulkanFFTTransfer vulkanFFTTransferSearchDir;
		void create() {}
		void init(regionbook regions_book, intfield regions, field<float> transformed_dipole_matrices, std::shared_ptr<Data::Geometry> geometry, int solver_type) {
			SIZES[0] = geometry->n_cells[0];
			SIZES[1] = geometry->n_cells[1];
			SIZES[2] = geometry->n_cells[2];
			LOCAL_SIZE[0] = std::min(SIZES[0] + 1, (uint32_t)513);
			LOCAL_SIZE[1] = 1;
			LOCAL_SIZE[2] = 1;
			LOCAL_SIZE_FFT[0] = 1;
			LOCAL_SIZE_FFT[1] = std::min(SIZES[1] + 1, (uint32_t)513);
			LOCAL_SIZE_FFT[2] = 1;
			vulkanFFTPlan.axes[0].sampleCount = 2 * SIZES[0];
			vulkanFFTPlan.axes[1].sampleCount = 2 * SIZES[1];
			vulkaniFFTPlan.axes[0].sampleCount = 2 * SIZES[0];
			vulkaniFFTPlan.axes[1].sampleCount = 2 * SIZES[1];
			if (SIZES[2] > 1) {
				vulkanFFTPlan.axes[2].sampleCount = 2 * SIZES[2];
				vulkaniFFTPlan.axes[2].sampleCount = 2 * SIZES[2];
			}
			else {
				vulkanFFTPlan.axes[2].sampleCount = SIZES[2];
				vulkaniFFTPlan.axes[2].sampleCount = SIZES[2];
			}
			// Buffer size of the storage buffer that will contain the rendered mandelbrot set.
			bufferSizeSpins = 3 * sizeof(float) * SIZES[0] * SIZES[1] * SIZES[2];
			bufferSizeGradient = 3 * sizeof(float) * SIZES[0] * SIZES[1] * SIZES[2];
			bufferSizeRegions_Book = sizeof(Regionvalues) * 10;
			bufferSizeRegions = sizeof(int) * SIZES[0] * SIZES[1] * SIZES[2];
			bufferSizeKernel = 6 * sizeof(FFT_cpx_type) * (vulkanFFTPlan.axes[0].sampleCount / 2 + 1) * (vulkanFFTPlan.axes[1].sampleCount) * (vulkanFFTPlan.axes[2].sampleCount);

			//bufferSizeSpins_FFT = 3 * 4 * sizeof(FFT::FFT_cpx_type) * SIZES[0] * SIZES[1] * SIZES[2];

			uboSizeDimensions = sizeof(VulkanDimensions);
			// Initialize vulkan:
			createInstance();
			findPhysicalDevice();
			createDevice();
			VkCommandPoolCreateInfo commandPoolCreateInfo = {};
			commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			// the queue family of this command pool. All command buffers allocated from this command pool,
			// must be submitted to queues of this family ONLY. 
			commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
			vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool);
			context.commandPool = commandPool;
			createBufferFFT(&context, &bufferSpins, &bufferMemorySpins, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeSpins);
			createBufferFFT(&context, &bufferSpinsInit, &bufferMemorySpinsInit, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeSpins);

			createBufferFFT(&context, &bufferGradient, &bufferMemoryGradient, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeGradient);
			createBufferFFT(&context, &bufferGradientOut, &bufferMemoryGradientOut, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeGradient);
			createBufferFFT(&context, &bufferRegions_Book, &bufferMemoryRegions_Book, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeRegions_Book);
			createBufferFFT(&context, &bufferRegions, &bufferMemoryRegions, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeRegions);
			createBufferFFT(&context, &kernel, &bufferMemoryKernel, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeKernel);
			createBufferFFT(&context, &uboDimensions, &uboMemoryDimensions, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, uboSizeDimensions);

			//createBuffer(&bufferSpins, &bufferMemorySpins, bufferSizeSpins);
			//createBuffer(&bufferGradient, &bufferMemoryGradient, bufferSizeGradient);
			//createBuffer(&bufferRegions_Book, &bufferMemoryRegions_Book, bufferSizeRegions_Book);
			//createBuffer(&bufferRegions, &bufferMemoryRegions, bufferSizeRegions);
			//createBuffer(&kernel, &bufferMemoryKernel, bufferSizeKernel);
			//createBuffer(&Spins_FFT, &bufferMemorySpins_FFT, bufferSizeSpins_FFT);
			//createBufferUBO(&uboDimensions, &uboMemoryDimensions, uboSizeDimensions);

			/*createDescriptorSetLayout();
			createDescriptorSet();
			createComputePipeline();
			createCommandBuffer();
			createDescriptorSetLayoutConvolution();
			createDescriptorSetConvolution();
			createComputePipelineConvolution();
			createCommandBufferConvolution();*/


			initVulkanFFTContext(&context);
			createVulkanFFT(&vulkanFFTPlan);

			initReduceBuffers();
			createReduce(&vulkanReduce.collection);
			vkMapMemory(vulkanLBFGS.context->device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1], 0, 3 * sizeof(scalar), 0, &mapReduce);
			switch (solver_type) {
			case 0:
				initLBFGS();
				break;
			case 1:
				initVP();
				break;
			}

			createFillZero(&collectionFillZero, &vulkanFFTPlan);
			createZeroPadding(&collectionZeroPadding, &vulkanFFTPlan);

			commandBufferFFT = createCommandBufferFFT(&context, NULL);
			recordVulkanFFT(&vulkanFFTPlan, commandBufferFFT);
			vkEndCommandBuffer(commandBufferFFT);


			createConvolution(&collectionConvolution, &vulkanFFTPlan);
			//createConvolutionHermitian(&vulkanFFTPlan);

			createVulkaniFFT(&vulkaniFFTPlan, &vulkanFFTPlan);
			commandBufferiFFT = createCommandBufferFFT(&context, NULL);
			recordVulkaniFFT(&vulkaniFFTPlan, commandBufferiFFT);
			vkEndCommandBuffer(commandBufferiFFT);
			createZeroPaddingRemove(&collectionZeroPaddingRemove, &vulkaniFFTPlan);
			createComputeGradients_noDDI(&collectionGradients_noDDI, &vulkanFFTPlan);

			createReadSpins(&collectionReadSpins, &vulkanFFTPlan);
			createWriteGradient(&collectionWriteGradient, &vulkanFFTPlan);
			createWriteSpins(&collectionWriteSpins, &vulkanFFTPlan);
			switch (solver_type) {
			case 0:
				createLBFGS(&vulkanLBFGS);
				break;
			case 1:
				createVP(&vulkanLBFGS);
				break;
			}
			/*commandBufferCollected[0] = commandBufferZeroPadding;
			commandBufferCollected[1] = commandBufferFFT;
			commandBufferCollected[2] = commandBufferConvolution;
			commandBufferCollected[3] = commandBufferConvolutionHermitian;
			commandBufferCollected[4] = commandBufferiFFT;
			commandBufferCollected[5] = commandBufferZeroPaddingRemove;
			commandBufferCollected[6] = commandBufferAll;
			submitInfoCollected.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfoCollected.commandBufferCount = 7; // submit a single command buffer
			submitInfoCollected.pCommandBuffers = commandBufferCollected; // the command buffer to submit.*/


			VulkanFFTTransfer vulkanFFTTransfer;
			vulkanFFTTransfer.context = &context;

			vulkanFFTTransfer.size = bufferSizeKernel;
			vulkanFFTTransfer.deviceBuffer = kernel;

			float* data = (float*)createVulkanFFTUpload(&vulkanFFTTransfer);
			for (uint32_t i = 0; i < 2*6 * (vulkanFFTPlan.axes[0].sampleCount / 2 + 1) * (vulkanFFTPlan.axes[1].sampleCount) * (vulkanFFTPlan.axes[2].sampleCount); ++i) {
				data[i] = transformed_dipole_matrices[i];
			}
			freeVulkanFFTTransfer(&vulkanFFTTransfer);

			vulkanFFTTransfer.size = sizeof(VulkanDimensions);
			vulkanFFTTransfer.deviceBuffer = uboDimensions;

			VulkanDimensions* ubo = (VulkanDimensions*)createVulkanFFTUpload(&vulkanFFTTransfer);
			ubo->WIDTH = SIZES[0];
			ubo->HEIGHT = SIZES[1];
			ubo->DEPTH = SIZES[2];
			ubo->n = SIZES[0] * SIZES[1] * SIZES[2];
			ubo->npad = (vulkanFFTPlan.axes[0].sampleCount) * (vulkanFFTPlan.axes[1].sampleCount / 2);
			ubo->npad2 = (vulkanFFTPlan.axes[0].sampleCount / 2 + 1) * (vulkanFFTPlan.axes[1].sampleCount);
			freeVulkanFFTTransfer(&vulkanFFTTransfer);

			vulkanFFTTransfer.size = bufferSizeRegions;
			vulkanFFTTransfer.deviceBuffer = bufferRegions;

			int* data_regions = (int*)createVulkanFFTUpload(&vulkanFFTTransfer);
			for (uint32_t i = 0; i < SIZES[0] * SIZES[1] * SIZES[2]; ++i) {
				data_regions[i] = regions[i];
			}
			freeVulkanFFTTransfer(&vulkanFFTTransfer);

			vulkanFFTTransfer.size = bufferSizeRegions_Book;
			vulkanFFTTransfer.deviceBuffer = bufferRegions_Book;

			Regionvalues* data_regions_book = (Regionvalues*)createVulkanFFTUpload(&vulkanFFTTransfer);
			for (uint32_t i = 0; i < 10; ++i) {
				data_regions_book[i] = regions_book[i];
			}
			freeVulkanFFTTransfer(&vulkanFFTTransfer);


			vulkanFFTTransferGradient.context = &context;
			vulkanFFTTransferGradient.size = bufferSizeGradient;
			vulkanFFTTransferGradient.deviceBuffer = bufferGradientOut;
			createBufferFFT(vulkanFFTTransferGradient.context, &vulkanFFTTransferGradient.hostBuffer, &vulkanFFTTransferGradient.deviceMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, vulkanFFTTransferGradient.size);
			//bufferTransfer(vulkanFFTTransferGradient.context, vulkanFFTTransferGradient.hostBuffer, vulkanFFTTransferGradient.deviceBuffer, vulkanFFTTransferGradient.size);
			//vulkanFFTTransferGradient.deviceBuffer = VK_NULL_HANDLE;


			vulkanFFTTransferSpins.context = &context;
			//vulkanFFTTransferSpins.size = vulkanFFTPlan.bufferSize;
			//vulkanFFTTransferSpins.deviceBuffer = vulkanFFTPlan.buffer[0];
			vulkanFFTTransferSpins.size = bufferSizeSpins;
			vulkanFFTTransferSpins.deviceBuffer = bufferSpinsInit;
			createBufferFFT(vulkanFFTTransferSpins.context, &vulkanFFTTransferSpins.hostBuffer, &vulkanFFTTransferSpins.deviceMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vulkanFFTTransferSpins.size);

			vulkanFFTTransferReduce.context = &context;
			vulkanFFTTransferReduce.size = 3 * sizeof(float);
			vulkanFFTTransferReduce.deviceBuffer = vulkanReduce.buffer[vulkanReduce.bufferNum - 1];
			createBufferFFT(vulkanFFTTransferReduce.context, &vulkanFFTTransferReduce.hostBuffer, &vulkanFFTTransferReduce.deviceMemory, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, vulkanFFTTransferReduce.size);

			vulkanFFTTransferReduceDot.context = &context;
			vulkanFFTTransferReduceDot.size = sizeof(float);
			vulkanFFTTransferReduceDot.deviceBuffer = vulkanReduce.buffer[vulkanReduce.bufferNum - 1];
			createBufferFFT(vulkanFFTTransferReduceDot.context, &vulkanFFTTransferReduceDot.hostBuffer, &vulkanFFTTransferReduceDot.deviceMemory, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, vulkanFFTTransferReduceDot.size);

			switch (solver_type) {
			case 0:
				vulkanFFTTransferGrad.context = &context;
				vulkanFFTTransferGrad.size = vulkanLBFGS.bufferSizes[7];
				vulkanFFTTransferGrad.deviceBuffer = vulkanLBFGS.buffer[7];
				createBufferFFT(vulkanFFTTransferGrad.context, &vulkanFFTTransferGrad.hostBuffer, &vulkanFFTTransferGrad.deviceMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vulkanFFTTransferGrad.size);

				vulkanFFTTransferSearchDir.context = &context;
				vulkanFFTTransferSearchDir.size = vulkanLBFGS.bufferSizes[6];
				vulkanFFTTransferSearchDir.deviceBuffer = vulkanLBFGS.buffer[6];
				createBufferFFT(vulkanFFTTransferSearchDir.context, &vulkanFFTTransferSearchDir.hostBuffer, &vulkanFFTTransferSearchDir.deviceMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, vulkanFFTTransferSearchDir.size);

				CreateBufferTransferSolver(vulkanLBFGS.context, vulkanLBFGS.buffer[8], vulkanLBFGS.buffer[7], vulkanLBFGS.bufferSizes[7], &commandBufferTransferSolver);
				CreateBufferTransferSolver(vulkanLBFGS.context, vulkanLBFGS.buffer[9], vulkanLBFGS.buffer[7], vulkanLBFGS.bufferSizes[7], &commandBufferTransferSolver2);
				break;
			case 1:
				vulkanFFTTransferGrad.context = &context;
				vulkanFFTTransferGrad.size = vulkanLBFGS.bufferSizes[1];
				vulkanFFTTransferGrad.deviceBuffer = vulkanLBFGS.buffer[1];
				createBufferFFT(vulkanFFTTransferGrad.context, &vulkanFFTTransferGrad.hostBuffer, &vulkanFFTTransferGrad.deviceMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vulkanFFTTransferGrad.size);

				vulkanFFTTransferSearchDir.context = &context;
				vulkanFFTTransferSearchDir.size = vulkanLBFGS.bufferSizes[0];
				vulkanFFTTransferSearchDir.deviceBuffer = vulkanLBFGS.buffer[0];
				createBufferFFT(vulkanFFTTransferSearchDir.context, &vulkanFFTTransferSearchDir.hostBuffer, &vulkanFFTTransferSearchDir.deviceMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, vulkanFFTTransferSearchDir.size);

				CreateBufferTransferSolver(vulkanLBFGS.context, vulkanLBFGS.buffer[2], vulkanLBFGS.buffer[1], vulkanLBFGS.bufferSizes[1], &commandBufferTransferSolver);
				break;
			}

			/*for (int i = 0; i < 6 * 2 * (SIZES[0] + 1) * SIZES[1]; i++) {
				std::cout << i << " " << transformed_dipole_matrices[i][0] << " " << transformed_dipole_matrices[i][1] << " kernel\n";
				//std::cout << i << " " << real[i][1] << " " << imag[i][1] << " 1 fft\n";
				//std::cout << i << " " << real[i][2] << " " << imag[i][2] << " 2 fft\n";
				//std::cout << spins[16 + 32 * 64][index] << " " << real[16 + 32 * 64][index] << " " << imag[16 + 32 * 64][index] << " backward\n";
			}*/
			/*FFT::FFT_cpx_type tt[6];
			vkMapMemory(device, bufferMemoryKernel, 0, bufferSizeKernel, 0, &mappedMemory);

			memcpy(tt, mappedMemory, sizeof(FFT::FFT_cpx_type) * 6);
			// Done reading, so unmap.
			vkUnmapMemory(device, bufferMemoryKernel);
			std::cout << transformed_dipole_matrices[0][0] << " " << transformed_dipole_matrices[1][0] << " " << transformed_dipole_matrices[2][0] << "\n";
			std::cout << tt[0][0] << " " << tt[1][0] << " " << tt[2][0] << "\n";*/

			//std::cout << "createPipe\n";
			//loadInput(regions_book, regions);
			//std::cout << "create\n";
			//std::cout << "save\n";
			// Clean up all vulkan resources.
			//cleanup();
		}
		void readDataStream(VulkanFFTTransfer vulkanFFTTransfer, const vectorfield& in) {

			void* map;
			vkMapMemory(device, vulkanFFTTransfer.deviceMemory, 0, vulkanFFTTransfer.size, 0, &map);

			/*FFT_cpx_type* data = (FFT_cpx_type*)(map);

			for (uint32_t z = 0; z < SIZES[2]; ++z) {
				uint32_t zOffset_zeropad = 4 * SIZES[0] * SIZES[1] * z;
				uint32_t zOffset_nonpad = SIZES[0] * SIZES[1] * z;
				for (uint32_t y = 0; y < SIZES[1]; ++y) {
					uint32_t yzOffset_zeropad = zOffset_zeropad + 2*SIZES[0] * y;
					uint32_t yzOffset_nonpad = zOffset_nonpad + SIZES[0] * y;
					for (uint32_t x = 0; x < SIZES[0]; ++x) {
						for (uint32_t index = 0; index < 3; index++) {
							data[3 * (x + yzOffset_zeropad) + index][0] = spins[x+yzOffset_nonpad][index];
						}
					}
				}
			}*/
			memcpy(map, in.data(), vulkanFFTTransfer.size);
			bufferTransfer(vulkanFFTTransfer.context, vulkanFFTTransfer.deviceBuffer, vulkanFFTTransfer.hostBuffer, vulkanFFTTransfer.size);
			vkUnmapMemory(device, vulkanFFTTransfer.deviceMemory);

		}
		void readInitialSpins(void* in) {
			void* map;
			vkMapMemory(device, vulkanFFTTransferSpins.deviceMemory, 0, vulkanFFTTransferSpins.size, 0, &map);

			/*FFT_cpx_type* data = (FFT_cpx_type*)(map);

			for (uint32_t z = 0; z < SIZES[2]; ++z) {
				uint32_t zOffset_zeropad = 4 * SIZES[0] * SIZES[1] * z;
				uint32_t zOffset_nonpad = SIZES[0] * SIZES[1] * z;
				for (uint32_t y = 0; y < SIZES[1]; ++y) {
					uint32_t yzOffset_zeropad = zOffset_zeropad + 2*SIZES[0] * y;
					uint32_t yzOffset_nonpad = zOffset_nonpad + SIZES[0] * y;
					for (uint32_t x = 0; x < SIZES[0]; ++x) {
						for (uint32_t index = 0; index < 3; index++) {
							data[3 * (x + yzOffset_zeropad) + index][0] = spins[x+yzOffset_nonpad][index];
						}
					}
				}
			}*/
			memcpy(map, in, vulkanFFTTransferSpins.size);
			bufferTransfer(vulkanFFTTransferSpins.context, vulkanFFTTransferSpins.deviceBuffer, vulkanFFTTransferSpins.hostBuffer, vulkanFFTTransferSpins.size);
			vkUnmapMemory(device, vulkanFFTTransferSpins.deviceMemory);
			runCommandBuffer(collectionReadSpins.commandBuffer);
		}

		void writeDataStream(VulkanFFTTransfer vulkanFFTTransfer, void* out) {
			/*VulkanFFTTransfer vulkanFFTTransfer;
			vulkanFFTTransfer.context = &context;
			vulkanFFTTransfer.size = size;
			vulkanFFTTransfer.deviceBuffer = buffer[0];
			auto data = reinterpret_cast<std::complex<float>*>(createVulkanFFTDownload(&vulkanFFTTransfer));
			for (uint32_t z = 0; z < SIZES[2]; ++z) {
				uint32_t zOffset_zeropad = 4 * SIZES[0] * SIZES[1] * z;
				uint32_t zOffset_nonpad = SIZES[0] * SIZES[1] * z;
				for (uint32_t y = 0; y < SIZES[1]; ++y) {
					uint32_t yzOffset_zeropad = zOffset_zeropad + 2 * SIZES[0] * y;
					uint32_t yzOffset_nonpad = zOffset_nonpad + SIZES[0] * y;
					for (uint32_t x = 0; x < SIZES[0]; ++x) {
						for (uint32_t index = 0; index < 3; index++) {
							gradient[x + yzOffset_nonpad][index]=data[3 * (x + yzOffset_zeropad) + index].real;
						}
					}
				}
			}

			freeVulkanFFTTransfer(&vulkanFFTTransfer);*/
			//createBufferFFT(vulkanFFTTransfer->context, &vulkanFFTTransfer->hostBuffer, &vulkanFFTTransfer->deviceMemory, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vulkanFFTTransfer->size);
			//auto time0 = std::chrono::steady_clock::now();
			//saveData(gradient);
			runCommandBuffer(collectionWriteGradient.commandBuffer);
			bufferTransfer(vulkanFFTTransfer.context, vulkanFFTTransfer.hostBuffer, vulkanFFTTransfer.deviceBuffer, vulkanFFTTransfer.size);
			//auto time1 = std::chrono::steady_clock::now();
			void* map;
			vkMapMemory(vulkanFFTTransfer.context->device, vulkanFFTTransfer.deviceMemory, 0, vulkanFFTTransfer.size, 0, &map);

			//auto time11 = std::chrono::steady_clock::now();
			//float* data = (float*)(map);
			memcpy(out, map, vulkanFFTTransfer.size);
			/*for (uint32_t i = 0; i < SIZES[0] * SIZES[1] * SIZES[2]; ++i) {
				for (uint32_t index = 0; index < 3; index++) {
					gradient[i][index] = data[3 * (i)+index];
				}
			}
			for (uint32_t z = 0; z < SIZES[2]; ++z) {
				uint32_t zOffset = SIZES[0] * SIZES[1] * z;
				for (uint32_t y = 0; y < SIZES[1]; ++y) {
					uint32_t yzOffset = zOffset + SIZES[0] * y;
					for (uint32_t x = 0; x < SIZES[0]; ++x) {
						for (uint32_t index = 0; index < 3; index++) {
							gradient[x + yzOffset][index] = data[3 * (x + yzOffset) + index];
						}
					}
				}
			}*/
			//auto time22 = std::chrono::steady_clock::now();
			//bufferTransfer(vulkanFFTTransfer.context, vulkanFFTTransfer.deviceBuffer, vulkanFFTTransfer.hostBuffer, vulkanFFTTransfer.size);
			vkUnmapMemory(device, vulkanFFTTransfer.deviceMemory);
			//auto time2 = std::chrono::steady_clock::now();
			//printf("buffertransfer: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() * 0.001);
			//printf("map: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time11 - time1).count() * 0.001);
			//printf("memcpy: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time22 - time11).count() * 0.001);
			//printf("unmap: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time2 - time22).count() * 0.001);
			//vkDestroyBuffer(device, vulkanFFTTransfer.hostBuffer, vulkanFFTTransfer->context->allocator);
			//vkFreeMemory(device, vulkanFFTTransfer->deviceMemory, vulkanFFTTransfer->context->allocator);
		}

		void writeSpins(void* out) {
			runCommandBuffer(collectionWriteSpins.commandBuffer);
			bufferTransfer(vulkanFFTTransferSpins.context, vulkanFFTTransferSpins.hostBuffer, vulkanFFTTransferSpins.deviceBuffer, vulkanFFTTransferSpins.size);
			//auto time1 = std::chrono::steady_clock::now();
			void* map;
			vkMapMemory(vulkanFFTTransferSpins.context->device, vulkanFFTTransferSpins.deviceMemory, 0, vulkanFFTTransferSpins.size, 0, &map);

			//auto time11 = std::chrono::steady_clock::now();
			//float* data = (float*)(map);
			memcpy(out, map, vulkanFFTTransferSpins.size);

			vkUnmapMemory(device, vulkanFFTTransferSpins.deviceMemory);

		}
		void writeGradient(void* out) {
			runCommandBuffer(collectionWriteGradient.commandBuffer);

			bufferTransfer(vulkanFFTTransferGradient.context, vulkanFFTTransferGradient.hostBuffer, vulkanFFTTransferGradient.deviceBuffer, vulkanFFTTransferGradient.size);
			//auto time1 = std::chrono::steady_clock::now();
			void* map;
			vkMapMemory(vulkanFFTTransferGradient.context->device, vulkanFFTTransferGradient.deviceMemory, 0, vulkanFFTTransferGradient.size, 0, &map);

			//auto time11 = std::chrono::steady_clock::now();
			//float* data = (float*)(map);
			memcpy(out, map, vulkanFFTTransferGradient.size);

			vkUnmapMemory(device, vulkanFFTTransferGradient.deviceMemory);


		}
		void initVulkanFFTContext(VulkanFFTContext* context) {
			vkGetPhysicalDeviceProperties(context->physicalDevice, &context->physicalDeviceProperties);
			vkGetPhysicalDeviceMemoryProperties(context->physicalDevice, &context->physicalDeviceMemoryProperties);

			VkFenceCreateInfo fenceCreateInfo = { };
			fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceCreateInfo.flags = 0;
			vkCreateFence(context->device, &fenceCreateInfo, context->allocator, &context->fence);
			//char filenames[SUPPORTED_RADIX_LEVELS][33];
			char filenames[1][40];
			sprintf(filenames[0], "shaders/radix_fft/fft_radix2_shared.spv");
			//sprintf(filenames[1], "shaders/radix_fft/fft_radix4.spv");
			//sprintf(filenames[2], "shaders/radix_fft/fft_radix8.spv");
			for (uint32_t i = 0; i < SUPPORTED_RADIX_LEVELS; ++i) {
				uint32_t filelength;
				// the code in comp.spv was created by running the command:
				// glslangValidator.exe -V shader.comp
				uint32_t* code = readFile(filelength, filenames[i]);
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &context->shaderModules[i]);
				delete[] code;
			}
			context->uboAlignment = context->physicalDeviceProperties.limits.minUniformBufferOffsetAlignment;
		}

		void createVulkanFFT(VulkanFFTPlan* vulkanFFTPlan) {
			vulkanFFTPlan->resultInSwapBuffer = false;
			vulkanFFTPlan->bufferSize = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * 2);
			vulkanFFTPlan->bufferSize[0] = 3 * sizeof(float) * 2 * vulkanFFTPlan->axes[0].sampleCount * (vulkanFFTPlan->axes[1].sampleCount / 2) * vulkanFFTPlan->axes[2].sampleCount;
			vulkanFFTPlan->bufferSize[1] = 3 * sizeof(float) * 2 * (vulkanFFTPlan->axes[0].sampleCount / 2 + 1) * vulkanFFTPlan->axes[1].sampleCount * vulkanFFTPlan->axes[2].sampleCount;
			vulkanFFTPlan->buffer = (VkBuffer*)malloc(sizeof(VkBuffer) * 2);
			vulkanFFTPlan->deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) * 2);
			for (uint32_t i = 0; i < 2; ++i) {
				createBufferFFT(vulkanFFTPlan->context, &vulkanFFTPlan->buffer[i], &vulkanFFTPlan->deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanFFTPlan->bufferSize[i]);
			}
			for (uint32_t i = 0; i < COUNT_OF(vulkanFFTPlan->axes); ++i) {
				if (vulkanFFTPlan->axes[i].sampleCount > 1)
					planVulkanFFTAxis(vulkanFFTPlan, i);

			}
		}

		void createVulkaniFFT(VulkanFFTPlan* vulkaniFFTPlan, VulkanFFTPlan* vulkanFFTPlan) {
			vulkaniFFTPlan->resultInSwapBuffer = vulkanFFTPlan->resultInSwapBuffer;
			vulkaniFFTPlan->bufferSize = vulkanFFTPlan->bufferSize;
			//for (uint32_t i = 0; i < COUNT_OF(vulkaniFFTPlan->buffer); ++i) {
			vulkaniFFTPlan->buffer = vulkanFFTPlan->buffer;
			vulkaniFFTPlan->deviceMemory = vulkanFFTPlan->deviceMemory;
			//}
			vulkaniFFTPlan->inverse = true;
			//const uint32_t remap[3] = { 1, 0, 2 };
			for (uint32_t i = 0; i < COUNT_OF(vulkaniFFTPlan->axes); ++i) {
				if (vulkaniFFTPlan->axes[i].sampleCount > 1)
					planVulkanFFTAxis(vulkaniFFTPlan, i);
			}
		}
		void createBufferFFT(VulkanFFTContext* context, VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usage, VkMemoryPropertyFlags propertyFlags, VkDeviceSize size) {
			uint32_t queueFamilyIndices[1] = {};
			VkBufferCreateInfo bufferCreateInfo = {};
			bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			bufferCreateInfo.queueFamilyIndexCount = COUNT_OF(queueFamilyIndices);
			bufferCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
			bufferCreateInfo.size = size;
			bufferCreateInfo.usage = usage;
			vkCreateBuffer(context->device, &bufferCreateInfo, context->allocator, buffer);
			VkMemoryRequirements memoryRequirements;
			vkGetBufferMemoryRequirements(context->device, *buffer, &memoryRequirements);
			VkMemoryAllocateInfo memoryAllocateInfo = {};
			memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			memoryAllocateInfo.allocationSize = memoryRequirements.size;
			memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, propertyFlags);
			vkAllocateMemory(context->device, &memoryAllocateInfo, context->allocator, deviceMemory);
			vkBindBufferMemory(context->device, *buffer, *deviceMemory, 0);

		}

		void* createVulkanFFTUpload(VulkanFFTTransfer* vulkanFFTTransfer) {
			createBufferFFT(vulkanFFTTransfer->context, &vulkanFFTTransfer->hostBuffer, &vulkanFFTTransfer->deviceMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vulkanFFTTransfer->size);
			void* map;
			vkMapMemory(device, vulkanFFTTransfer->deviceMemory, 0, vulkanFFTTransfer->size, 0, &map);
			return map;
		}

		void* createVulkanFFTDownload(VulkanFFTTransfer* vulkanFFTTransfer) {
			createBufferFFT(vulkanFFTTransfer->context, &vulkanFFTTransfer->hostBuffer, &vulkanFFTTransfer->deviceMemory, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vulkanFFTTransfer->size);
			bufferTransfer(vulkanFFTTransfer->context, vulkanFFTTransfer->hostBuffer, vulkanFFTTransfer->deviceBuffer, vulkanFFTTransfer->size);
			vulkanFFTTransfer->deviceBuffer = VK_NULL_HANDLE;
			void* map;
			vkMapMemory(vulkanFFTTransfer->context->device, vulkanFFTTransfer->deviceMemory, 0, vulkanFFTTransfer->size, 0, &map);
			return map;
		}
		VkCommandBuffer createCommandBufferFFT(VulkanFFTContext* context, VkCommandBufferUsageFlags usageFlags) {
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = context->commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VkCommandBuffer commandBuffer;
			vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
			VkCommandBufferBeginInfo commandBufferBeginInfo = {};
			commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			commandBufferBeginInfo.flags = usageFlags;
			vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
			return commandBuffer;
		}
		void bufferTransfer(VulkanFFTContext* context, VkBuffer dstBuffer, VkBuffer srcBuffer, VkDeviceSize size) {
			VkCommandBuffer commandBuffer = createCommandBufferFFT(context, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
			VkBufferCopy copyRegion = {};
			copyRegion.srcOffset = 0;
			copyRegion.dstOffset = 0;
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
			vkEndCommandBuffer(commandBuffer);
			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(context->queue, 1, &submitInfo, context->fence);
			vkWaitForFences(context->device, 1, &context->fence, VK_TRUE, 100000000000);
			vkResetFences(context->device, 1, &context->fence);
			vkFreeCommandBuffers(context->device, context->commandPool, 1, &commandBuffer);
		}
		void CreateBufferTransferSolver(VulkanFFTContext* context, VkBuffer dstBuffer, VkBuffer srcBuffer, VkDeviceSize size, VkCommandBuffer* commandBuffer) {
			commandBuffer[0] = createCommandBufferFFT(context, NULL);
			VkBufferCopy copyRegion = {};
			copyRegion.srcOffset = 0;
			copyRegion.dstOffset = 0;
			copyRegion.size = size;
			vkCmdCopyBuffer(commandBuffer[0], srcBuffer, dstBuffer, 1, &copyRegion);
			vkEndCommandBuffer(commandBuffer[0]);

		}
		void bufferTransferSolver(VulkanFFTContext* context, VkCommandBuffer* commandBuffer) {

			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = commandBuffer;
			vkQueueSubmit(context->queue, 1, &submitInfo, context->fence);
			vkWaitForFences(context->device, 1, &context->fence, VK_TRUE, 100000000000);
			vkResetFences(context->device, 1, &context->fence);
		}
		void freeVulkanFFTTransfer(VulkanFFTTransfer* vulkanFFTTransfer) {
			if (vulkanFFTTransfer->deviceBuffer)
				bufferTransfer(vulkanFFTTransfer->context, vulkanFFTTransfer->deviceBuffer, vulkanFFTTransfer->hostBuffer, vulkanFFTTransfer->size);
			vkUnmapMemory(device, vulkanFFTTransfer->deviceMemory);
			vkDestroyBuffer(device, vulkanFFTTransfer->hostBuffer, vulkanFFTTransfer->context->allocator);
			vkFreeMemory(device, vulkanFFTTransfer->deviceMemory, vulkanFFTTransfer->context->allocator);
		}
		void recordVulkanFFT(VulkanFFTPlan* vulkanFFTPlan, VkCommandBuffer commandBuffer) {
			const uint32_t remap[3][3] = { {0, 1, 2} ,{1, 0, 2}, {2, 0, 1} };
			bool bit_swap = vulkanFFTPlan->resultInSwapBuffer;
			/*for (uint32_t i = 0; i < COUNT_OF(vulkanFFTPlan->axes); ++i) {

				if (vulkanFFTPlan->axes[remap[i][0]].sampleCount <= 1)
					continue;*/
			VulkanFFTAxis* vulkanFFTAxis = &vulkanFFTPlan->axes[remap[0][0]];
			uint32_t dynamicOffset = 0;
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelines[30 - __builtin_clz(vulkanFFTAxis->stageRadix[0])]);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelineLayout, 0, 1, &vulkanFFTAxis->descriptorSets[bit_swap], 1, &dynamicOffset);
			vkCmdDispatch(commandBuffer, 1, (uint32_t)ceil(3 * (vulkanFFTPlan->axes[remap[0][1]].sampleCount / 2)), vulkanFFTPlan->axes[remap[0][2]].sampleCount);
			createR2C_decomp(&collectionR2C_decomp, vulkanFFTPlan);
			for (int i = 0; i < 3; i++) {
				uint32_t push_consts[2] = { i * vulkanFFTPlan->axes[0].sampleCount * vulkanFFTPlan->axes[1].sampleCount / 2, i * (vulkanFFTPlan->axes[0].sampleCount / 2 + 1) * vulkanFFTPlan->axes[1].sampleCount };
				uint32_t dims[3] = { 32,32,1 };
				vkCmdPushConstants(commandBuffer, collectionR2C_decomp.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t), push_consts);
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collectionR2C_decomp.pipelines[0]);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collectionR2C_decomp.pipelineLayout, 0, 1, collectionR2C_decomp.descriptorSets, 0, NULL);
				vkCmdDispatch(commandBuffer, (uint32_t)ceil((vulkanFFTPlan->axes[0].sampleCount / 2 + 1) / ((scalar)dims[0])), (uint32_t)ceil(vulkanFFTPlan->axes[1].sampleCount / 2 / ((scalar)dims[1])), (uint32_t)ceil(vulkanFFTPlan->axes[2].sampleCount / ((scalar)dims[2])));
			}
			vulkanFFTAxis = &vulkanFFTPlan->axes[remap[1][0]];
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelines[30 - __builtin_clz(vulkanFFTAxis->stageRadix[0])]);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelineLayout, 0, 1, &vulkanFFTAxis->descriptorSets[bit_swap], 1, &dynamicOffset);
			vkCmdDispatch(commandBuffer, 1, (uint32_t)ceil(3 * (vulkanFFTPlan->axes[remap[1][1]].sampleCount / 2 + 1)), vulkanFFTPlan->axes[remap[1][2]].sampleCount);

			if (vulkanFFTPlan->axes[remap[2][0]].sampleCount > 1) {
				vulkanFFTAxis = &vulkanFFTPlan->axes[remap[2][0]];

				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelines[30 - __builtin_clz(vulkanFFTAxis->stageRadix[0])]);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelineLayout, 0, 1, &vulkanFFTAxis->descriptorSets[bit_swap], 1, &dynamicOffset);
				vkCmdDispatch(commandBuffer, 1, (uint32_t)ceil(vulkanFFTPlan->axes[remap[2][1]].sampleCount / 2 + 1), vulkanFFTPlan->axes[remap[2][2]].sampleCount);

			}
			//}

		}
		void recordVulkaniFFT(VulkanFFTPlan* vulkanFFTPlan, VkCommandBuffer commandBuffer) {
			const uint32_t remap[3][3] = { {2, 0, 1}, {1, 0, 2}, {0, 1, 2} };
			bool bit_swap = vulkanFFTPlan->resultInSwapBuffer;
			/*for (uint32_t i = 0; i < COUNT_OF(vulkanFFTPlan->axes); ++i) {

				if (vulkanFFTPlan->axes[remap[i][0]].sampleCount <= 1)
					continue;*/
			VulkanFFTAxis* vulkanFFTAxis;
			uint32_t dynamicOffset = 0;
			if (vulkanFFTPlan->axes[remap[0][0]].sampleCount > 1) {
				vulkanFFTAxis = &vulkanFFTPlan->axes[remap[0][0]];
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelines[30 - __builtin_clz(vulkanFFTAxis->stageRadix[0])]);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelineLayout, 0, 1, &vulkanFFTAxis->descriptorSets[bit_swap], 1, &dynamicOffset);
				vkCmdDispatch(commandBuffer, 1, (uint32_t)ceil(vulkanFFTPlan->axes[remap[0][1]].sampleCount / 2 + 1), vulkanFFTPlan->axes[remap[0][2]].sampleCount);

			}

			vulkanFFTAxis = &vulkanFFTPlan->axes[remap[1][0]];
			
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelines[30 - __builtin_clz(vulkanFFTAxis->stageRadix[0])]);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelineLayout, 0, 1, &vulkanFFTAxis->descriptorSets[bit_swap], 1, &dynamicOffset);
			vkCmdDispatch(commandBuffer, 1, (uint32_t)ceil(3 * (vulkanFFTPlan->axes[remap[1][1]].sampleCount / 2 + 1)), vulkanFFTPlan->axes[remap[1][2]].sampleCount);

			createC2R_decomp(&collectionC2R_decomp, vulkanFFTPlan);
			for (int i = 0; i < 3; i++) {
				uint32_t push_consts[2] = { i * (vulkanFFTPlan->axes[0].sampleCount / 2 + 1) * vulkanFFTPlan->axes[1].sampleCount, i * vulkanFFTPlan->axes[0].sampleCount * vulkanFFTPlan->axes[1].sampleCount / 2 };
				uint32_t dims[3] = { 32,32,1 };
				vkCmdPushConstants(commandBuffer, collectionC2R_decomp.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t), push_consts);
				vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collectionC2R_decomp.pipelines[0]);
				vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collectionC2R_decomp.pipelineLayout, 0, 1, collectionC2R_decomp.descriptorSets, 0, NULL);
				vkCmdDispatch(commandBuffer, (uint32_t)ceil((vulkanFFTPlan->axes[1].sampleCount / 2) / ((scalar)dims[0])), (uint32_t)ceil((vulkanFFTPlan->axes[0].sampleCount / 2 + 1) / ((scalar)dims[1])), (uint32_t)ceil(vulkanFFTPlan->axes[2].sampleCount / ((scalar)dims[2])));
			}
			
			vulkanFFTAxis = &vulkanFFTPlan->axes[remap[2][0]];
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelines[30 - __builtin_clz(vulkanFFTAxis->stageRadix[0])]);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, vulkanFFTAxis->pipelineLayout, 0, 1, &vulkanFFTAxis->descriptorSets[bit_swap], 1, &dynamicOffset);

			vkCmdDispatch(commandBuffer, 1, (uint32_t)ceil(3 * (vulkanFFTPlan->axes[remap[2][1]].sampleCount / 2)), vulkanFFTPlan->axes[remap[2][2]].sampleCount);

		}

		void planVulkanFFTAxis(VulkanFFTPlan* vulkanFFTPlan, uint32_t axis) {
			const uint32_t remap[3][3] = { {0, 1, 2}, {1, 0, 2}, {2, 0, 1} };
			VulkanFFTAxis* vulkanFFTAxis = &vulkanFFTPlan->axes[axis];
			{
				vulkanFFTAxis->stageCount = 31 - __builtin_clz(vulkanFFTAxis->sampleCount); // Logarithm of base 2
				vulkanFFTAxis->stageRadix = (uint32_t*)malloc(sizeof(uint32_t) * vulkanFFTAxis->stageCount);
				uint32_t stageSize = vulkanFFTAxis->sampleCount;
				vulkanFFTAxis->stageCount = 0;
				while (stageSize > 1) {
					uint32_t radixIndex = SUPPORTED_RADIX_LEVELS;
					do {
						assert(radixIndex > 0);
						--radixIndex;
						vulkanFFTAxis->stageRadix[vulkanFFTAxis->stageCount] = 2 << radixIndex;
					} while (stageSize % vulkanFFTAxis->stageRadix[vulkanFFTAxis->stageCount] > 0);
					stageSize /= vulkanFFTAxis->stageRadix[vulkanFFTAxis->stageCount];
					++vulkanFFTAxis->stageCount;
				}
			}

			{

				vulkanFFTAxis->uboSize = vulkanFFTPlan->context->uboAlignment * 1;
				createBufferFFT(vulkanFFTPlan->context, &vulkanFFTAxis->ubo, &vulkanFFTAxis->uboDeviceMemory, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanFFTAxis->uboSize);
				VulkanFFTTransfer vulkanFFTTransfer;
				vulkanFFTTransfer.context = vulkanFFTPlan->context;
				vulkanFFTTransfer.size = vulkanFFTAxis->uboSize;
				vulkanFFTTransfer.deviceBuffer = vulkanFFTAxis->ubo;
				//std::cout << vulkanFFTAxis->uboSize << " " << vulkanFFTPlan->context->uboAlignment << " " << vulkanFFTAxis->ubo << "uu\n";
				char* ubo = (char*)createVulkanFFTUpload(&vulkanFFTTransfer);

				uint32_t strides[3] = { 1, vulkanFFTPlan->axes[remap[axis][0]].sampleCount, vulkanFFTPlan->axes[remap[axis][0]].sampleCount * vulkanFFTPlan->axes[remap[axis][1]].sampleCount };
				uint32_t stageSize = 1;
				for (uint32_t j = 0; j < 1; ++j) {
					VulkanFFTUBO* uboFrame = (VulkanFFTUBO*)&ubo[vulkanFFTPlan->context->uboAlignment * j];
					uboFrame->stride[0] = strides[0];
					uboFrame->stride[1] = strides[1];
					uboFrame->stride[2] = strides[2];
					uboFrame->radixStride = vulkanFFTAxis->sampleCount / vulkanFFTAxis->stageRadix[j];
					uboFrame->stageSize = stageSize;
					uboFrame->directionFactor = (vulkanFFTPlan->inverse) ? -1.0F : 1.0F;
					uboFrame->angleFactor = uboFrame->directionFactor * (float)(3.14159265358979 / uboFrame->stageSize);
					uboFrame->normalizationFactor = (vulkanFFTPlan->inverse) ? 1.0F : 1.0F / vulkanFFTAxis->stageRadix[j];
					uboFrame->stageCount = vulkanFFTAxis->stageCount;
					uboFrame->WIDTH = SIZES[0];
					uboFrame->HEIGHT = SIZES[1];
					uboFrame->DEPTH = SIZES[2];
					if (axis == 0) {
						uboFrame->npad = (vulkanFFTPlan->axes[0].sampleCount) * (vulkanFFTPlan->axes[1].sampleCount / 2);
					}
					else
					{
						uboFrame->npad = (vulkanFFTPlan->axes[0].sampleCount / 2 + 1) * (vulkanFFTPlan->axes[1].sampleCount);
					}

					stageSize *= vulkanFFTAxis->stageRadix[j];
				}
				freeVulkanFFTTransfer(&vulkanFFTTransfer);
			}

			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &vulkanFFTAxis->descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				vulkanFFTAxis->descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				vulkanFFTAxis->descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &vulkanFFTAxis->descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = vulkanFFTAxis->descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = vulkanFFTAxis->descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, vulkanFFTAxis->descriptorSets);
				for (uint32_t j = 0; j < 1; ++j)
					for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
						VkDescriptorBufferInfo descriptorBufferInfo = { };
						if (i == 0) {
							descriptorBufferInfo.buffer = vulkanFFTAxis->ubo;
							//descriptorBufferInfo.offset = vulkanFFTPlan->context->uboAlignment * j;
							descriptorBufferInfo.offset = 0;
							descriptorBufferInfo.range = sizeof(VulkanFFTUBO);
						}
						if (i == 1) {
							//descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[1 - (vulkanFFTPlan->resultInSwapBuffer + i + j) % 2];
							if (axis == 0) {
								if (vulkanFFTPlan->inverse == false) {
									descriptorBufferInfo.buffer = bufferSpins;
									descriptorBufferInfo.offset = 0;
									descriptorBufferInfo.range = bufferSizeSpins;
								}
								else {
									descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[0];
									descriptorBufferInfo.offset = 0;
									descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[0];
								}
							}
							else {
								descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[1];
								descriptorBufferInfo.offset = 0;
								descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[1];
							}

						}
						if (i == 2) {
							//descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[1 - (vulkanFFTPlan->resultInSwapBuffer + i + j) % 2];
							if (axis == 0) {
								if (vulkanFFTPlan->inverse == true) {
									descriptorBufferInfo.buffer = bufferGradient;
									descriptorBufferInfo.offset = 0;
									descriptorBufferInfo.range = bufferSizeGradient;
								}
								else {
									descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[0];
									descriptorBufferInfo.offset = 0;
									descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[0];
								}
							}
							else {
								descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[1];
								descriptorBufferInfo.offset = 0;
								descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[1];
							}

						}
						VkWriteDescriptorSet writeDescriptorSet = { };
						writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						writeDescriptorSet.dstSet = vulkanFFTAxis->descriptorSets[j];
						writeDescriptorSet.dstBinding = i;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorType = descriptorType[i];
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
						vkUpdateDescriptorSets(vulkanFFTPlan->context->device, 1, &writeDescriptorSet, 0, NULL);
					}
			}

			{
				vulkanFFTAxis->pipelines = (VkPipeline*)malloc(sizeof(VkPipeline) * SUPPORTED_RADIX_LEVELS);
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = vulkanFFTAxis->descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &vulkanFFTAxis->pipelineLayout);

				uint32_t fft_consts_init[5];
				fft_consts_init[0] = 128;
				fft_consts_init[1] = 1;
				fft_consts_init[2] = 1;
				fft_consts_init[3] = vulkanFFTAxis->sampleCount;
				if (axis == 0) {
					if (vulkanFFTPlan->inverse == true) {
						fft_consts_init[4] = 2;
					}
					else {
						fft_consts_init[4] = 0;
					}
				}
				else {
					fft_consts_init[4] = 1;
				}
				//std::cout << fft_consts_init[4] << "\n";
				std::array<VkSpecializationMapEntry, 5> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				specializationMapEntries[1].constantID = 2;
				specializationMapEntries[1].size = sizeof(uint32_t);
				specializationMapEntries[1].offset = sizeof(uint32_t);
				specializationMapEntries[2].constantID = 3;
				specializationMapEntries[2].size = sizeof(uint32_t);
				specializationMapEntries[2].offset = 2 * sizeof(uint32_t);
				specializationMapEntries[3].constantID = 4;
				specializationMapEntries[3].size = sizeof(uint32_t);
				specializationMapEntries[3].offset = 3 * sizeof(uint32_t);
				specializationMapEntries[4].constantID = 5;
				specializationMapEntries[4].size = sizeof(uint32_t);
				specializationMapEntries[4].offset = 4 * sizeof(uint32_t);
				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = 5 * sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &fft_consts_init;
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[SUPPORTED_RADIX_LEVELS] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[SUPPORTED_RADIX_LEVELS] = { };

				for (uint32_t i = 0; i < SUPPORTED_RADIX_LEVELS; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					pipelineShaderStageCreateInfo[i].module = vulkanFFTPlan->context->shaderModules[i];
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = vulkanFFTAxis->pipelineLayout;

				}

				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, SUPPORTED_RADIX_LEVELS, computePipelineCreateInfo, vulkanFFTPlan->context->allocator, vulkanFFTAxis->pipelines);

			}

			//	if (vulkanFFTAxis->stageCount %2 !=0)
				//	vulkanFFTPlan->resultInSwapBuffer = !vulkanFFTPlan->resultInSwapBuffer;
		}
		//void run(const vectorfield& spins, vectorfield& gradient) {
		void run() {
			bool timed = true;
			if (timed == true) {
				auto time0 = std::chrono::steady_clock::now();
				//loadData(spins);
				//std::cout << "load\n";
				// Finally, run the recorded command buffer.
				//std::cout << sizeof(float) << " " << sizeof(Vector3) << " " << sizeof(FFT::FFT_cpx_type) << "run\n";
				//runCommandBuffer(commandBufferAll);
				//saveData(gradient);

				//vectorfield real = vectorfield(4*SIZES[0] * SIZES[1], { 1.0,1.0,1.0 });
				//vectorfield imag = vectorfield(4 * SIZES[0] * SIZES[1], { 1.0,1.0,1.0 });
				//runCommandBuffer(collectionFillZero.commandBuffer);
				auto timeA = std::chrono::steady_clock::now();

				//runCommandBuffer(commandBufferFillZero);
				//runCommandBuffer(commandBufferFillZero);
				//runCommandBuffer(commandBufferFillZero);
				//vulkanFFTPlan.resultInSwapBuffer = false;
				//vulkaniFFTPlan.resultInSwapBuffer = false;
				//readDataStream(vulkanFFTTransferSpins, spins);


				//vkQueueSubmit(context.queue, 1, &submitInfoCollected, context.fence);
				//vkWaitForFences(context.device, 1, &context.fence, VK_TRUE, 100000000000);
				//vkResetFences(context.device, 1, &context.fence);
				auto timeB = std::chrono::steady_clock::now();
				//runCommandBuffer(collectionZeroPadding.commandBuffer);
				auto timeC = std::chrono::steady_clock::now();
				runCommandBuffer(commandBufferFFT);
				auto timeD = std::chrono::steady_clock::now();
				//writeDataStream(&vulkanFFTPlan, real, imag);

				//bufferTransfer(&context, vulkanFFTPlan.buffer[!vulkanFFTPlan.resultInSwapBuffer], vulkanFFTPlan.buffer[vulkanFFTPlan.resultInSwapBuffer], vulkanFFTPlan.bufferSize);
				runCommandBuffer(collectionConvolution.commandBuffer);
				auto timeE = std::chrono::steady_clock::now();
				//runCommandBuffer(commandBufferConvolutionHermitian);
				auto timeF = std::chrono::steady_clock::now();
				//bufferTransfer(&context, vulkaniFFTPlan.buffer[0], Spins_FFT, bufferSizeSpins_FFT);
				//vulkanFFTPlan.resultInSwapBuffer = false;
				runCommandBuffer(commandBufferiFFT);
				auto timeG = std::chrono::steady_clock::now();
				//runCommandBuffer(collectionZeroPaddingRemove.commandBuffer);

				//writeDataStream(&vulkaniFFTPlan, real, imag);
				auto timeH = std::chrono::steady_clock::now();
				runCommandBuffer(collectionGradients_noDDI.commandBuffer);
				auto timeK = std::chrono::steady_clock::now();
				//saveData(gradient);*/
				//writeDataStream(vulkanFFTTransferGradient, gradient);

				auto timeL = std::chrono::steady_clock::now();

				//printf("Fill0: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeA - time0).count() * 0.001);
				//printf("read: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeB - timeA).count() * 0.001);
				//printf("zeropad: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeC - timeB).count() * 0.001);
				printf("FFT: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeD - timeC).count() * 0.001);
				//printf( "Convo: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeE - timeD).count() * 0.001);
				//printf( "ConvoHermit: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeF - timeE).count() * 0.001);
				printf("iFFT: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeG - timeF).count() * 0.001);
				//printf("ZeroPadRemove: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeH - timeG).count() * 0.001);
				//printf("all: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeK - timeH).count() * 0.001);
				//printf("savedata: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeL - timeK).count() * 0.001);

				//printf("time: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeL - time0).count() * 0.001);
			}
			else {
				auto time0 = std::chrono::steady_clock::now();
				//loadData(spins);
				//std::cout << "load\n";
				// Finally, run the recorded command buffer.
				//std::cout << sizeof(float) << " " << sizeof(Vector3) << " " << sizeof(FFT::FFT_cpx_type) << "run\n";
				//runCommandBuffer(commandBufferAll);
				//saveData(gradient);

				//vectorfield real = vectorfield(4*SIZES[0] * SIZES[1], { 1.0,1.0,1.0 });
				//vectorfield imag = vectorfield(4 * SIZES[0] * SIZES[1], { 1.0,1.0,1.0 });
				//runCommandBuffer(collectionFillZero.commandBuffer);
				//auto timeA = std::chrono::steady_clock::now();

				//runCommandBuffer(commandBufferFillZero);
				//runCommandBuffer(commandBufferFillZero);
				//runCommandBuffer(commandBufferFillZero);
				//vulkanFFTPlan.resultInSwapBuffer = false;
				//vulkaniFFTPlan.resultInSwapBuffer = false;
				//readDataStream(vulkanFFTTransferSpins, spins);


				//vkQueueSubmit(context.queue, 1, &submitInfoCollected, context.fence);
				//vkWaitForFences(context.device, 1, &context.fence, VK_TRUE, 100000000000);
				//vkResetFences(context.device, 1, &context.fence);
				//auto timeB = std::chrono::steady_clock::now();
				//runCommandBuffer(collectionZeroPadding.commandBuffer);
				//auto timeC = std::chrono::steady_clock::now();
				runCommandBuffer(commandBufferFFT);
				//auto timeD = std::chrono::steady_clock::now();
				//writeDataStream(&vulkanFFTPlan, real, imag);

				//bufferTransfer(&context, vulkanFFTPlan.buffer[!vulkanFFTPlan.resultInSwapBuffer], vulkanFFTPlan.buffer[vulkanFFTPlan.resultInSwapBuffer], vulkanFFTPlan.bufferSize);
				runCommandBuffer(collectionConvolution.commandBuffer);
				//auto timeE = std::chrono::steady_clock::now();
				//runCommandBuffer(commandBufferConvolutionHermitian);
				//auto timeF = std::chrono::steady_clock::now();
				//bufferTransfer(&context, vulkaniFFTPlan.buffer[0], Spins_FFT, bufferSizeSpins_FFT);
				//vulkanFFTPlan.resultInSwapBuffer = false;
				runCommandBuffer(commandBufferiFFT);
				//auto timeG = std::chrono::steady_clock::now();
				//runCommandBuffer(collectionZeroPaddingRemove.commandBuffer);

				//writeDataStream(&vulkaniFFTPlan, real, imag);
				//auto timeH = std::chrono::steady_clock::now();
				runCommandBuffer(collectionGradients_noDDI.commandBuffer);
				//auto timeK = std::chrono::steady_clock::now();
				//saveData(gradient);*/
				//writeDataStream(vulkanFFTTransferGradient, gradient);
				auto timeL = std::chrono::steady_clock::now();

				/*printf("Fill0: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeA - time0).count() * 0.001);
				printf("read: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeB - timeA).count() * 0.001);
				printf("zeropad: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeC - timeB).count() * 0.001);
				printf( "FFT: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeD - timeC).count() * 0.001);
				printf( "Convo: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeE - timeD).count() * 0.001);
				printf( "ConvoHermit: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeF - timeE).count() * 0.001);
				printf("iFFT: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeG - timeF).count() * 0.001);
				printf("ZeroPadRemove: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeH - timeG).count() * 0.001);
				printf("all: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeK - timeH).count() * 0.001);
				printf("savedata: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeL - timeK).count() * 0.001);*/

				//printf("time: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeL - time0).count() * 0.001);
			}
		}
		scalar getEnergy() {
			scalar energy = 0;
			runCommandBuffer(collectionReduceDotEnergy.commandBuffer);
			if (SIZES[0] * SIZES[1] * SIZES[2] > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
			//if (SIZES[0] * SIZES[1] * SIZES[2] > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);

			auto time2 = std::chrono::steady_clock::now();
			//void* map;
			//vkMapMemory(vulkanLBFGS.context->device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1], 0, sizeof(scalar), 0, &mapReduce);
			memcpy(&energy, mapReduce, sizeof(scalar));
			//vkUnmapMemory(device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1]);
			return energy;
		}
		void oso_calc_gradients() { runCommandBuffer(vulkanLBFGS.collectionOsoCalcGradients.commandBuffer); }

		void scale(scalar maxmove) {
			int nos = SIZES[0] * SIZES[1] * SIZES[2];
			scalar theta_rms = 0;
			//theta_rms = sqrt(Backend::par::reduce(searchdir, [] SPIRIT_LAMBDA(const Vector3 & v) { return v.squaredNorm(); }) / nos);
			runCommandBuffer(vulkanLBFGS.collectionReduceDotScaling.commandBuffer);
			if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
			//if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);

			writeDataStream(vulkanFFTTransferReduceDot, &theta_rms);
			//std::cout << theta_rms << " thetarms\n";
			theta_rms = sqrt(theta_rms / nos);
			scalar scaling = (theta_rms > maxmove) ? maxmove / theta_rms : 1.0;
			//std::cout << theta_rms << " thetarms\n";
			//std::cout << scaling << " scaling\n";
			vulkanLBFGS.scaling = scaling;
			recordScale(&vulkanLBFGS.collectionScale);
			runCommandBuffer(vulkanLBFGS.collectionScale.commandBuffer);
		}
		void oso_rotate() {
			runCommandBuffer(vulkanLBFGS.collectionOsoRotate.commandBuffer);
		}
		void loadInput(regionbook regions_book, intfield regions) {
			void* mappedMemory = NULL;
			// Map the buffer memory, so that we can read from it on the CPU.
			vkMapMemory(device, bufferMemoryRegions_Book, 0, bufferSizeRegions_Book, 0, &mappedMemory);
			Regionvalues* pmappedMemory2 = (Regionvalues*)mappedMemory;
			// Get the color data from the buffer, and cast it to bytes.
			// We save the data to a vector.
			memcpy(pmappedMemory2, regions_book.data(), sizeof(Regionvalues) * 2);
			// Done reading, so unmap.
			vkUnmapMemory(device, bufferMemoryRegions_Book);
			//std::cout << "2\n";
			mappedMemory = NULL;
			// Map the buffer memory, so that we can read from it on the CPU.
			vkMapMemory(device, bufferMemoryRegions, 0, bufferSizeRegions, 0, &mappedMemory);
			int* pmappedMemory3 = (int*)mappedMemory;
			// Get the color data from the buffer, and cast it to bytes.
			// We save the data to a vector.
			memcpy(pmappedMemory3, regions.data(), sizeof(int) * SIZES[0] * SIZES[1] * SIZES[2]);
			// Done reading, so unmap.
			vkUnmapMemory(device, bufferMemoryRegions);
			//std::cout << "3\n";

		}
		void loadData(const vectorfield& spins) {
			void* mappedMemory = NULL;
			// Map the buffer memory, so that we can read from it on the CPU.
			vkMapMemory(device, bufferMemorySpins, 0, bufferSizeSpins, 0, &mappedMemory);
			Vector3* pmappedMemory0 = (Vector3*)mappedMemory;
			// Get the color data from the buffer, and cast it to bytes.
			// We save the data to a vector.
			memcpy(pmappedMemory0, spins.data(), 3 * sizeof(float) * SIZES[0] * SIZES[1] * SIZES[2]);
			// Done reading, so unmap.
			vkUnmapMemory(device, bufferMemorySpins);
			//std::cout << "0\n";
			/*mappedMemory = NULL;
			// Map the buffer memory, so that we can read from it on the CPU.
			vkMapMemory(device, bufferMemoryGradient, 0, bufferSizeGradient, 0, &mappedMemory);
			Vector3* pmappedMemory1 = (Vector3*)mappedMemory;
			// Get the color data from the buffer, and cast it to bytes.
			// We save the data to a vector.
			memcpy(pmappedMemory1, gradient.data(), 3 * sizeof(float) * SIZES[0] * SIZES[1] * SIZES[2]);
			// Done reading, so unmap.
			vkUnmapMemory(device, bufferMemoryGradient);*/

		}
		void saveData(vectorfield& gradient) {
			void* mappedMemory = NULL;
			// Map the buffer memory, so that we can read from it on the CPU.
			vkMapMemory(device, bufferMemoryGradient, 0, bufferSizeGradient, 0, &mappedMemory);
			Vector3* pmappedMemory1 = (Vector3*)mappedMemory;
			// Get the color data from the buffer, and cast it to bytes.
			// We save the data to a vector.
			memcpy(gradient.data(), pmappedMemory1, sizeof(Vector3) * SIZES[0] * SIZES[1] * SIZES[2]);
			// Done reading, so unmap.
			vkUnmapMemory(device, bufferMemoryGradient);
		}
		static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
			VkDebugReportFlagsEXT                       flags,
			VkDebugReportObjectTypeEXT                  objectType,
			uint64_t                                    object,
			size_t                                      location,
			int32_t                                     messageCode,
			const char* pLayerPrefix,
			const char* pMessage,
			void* pUserData) {

			printf("Debug Report: %s: %s\n", pLayerPrefix, pMessage);

			return VK_FALSE;
		}

		void createInstance() {
			std::vector<const char*> enabledExtensions;


			/*
			uint32_t layerCount;
			vkEnumerateInstanceLayerProperties(&layerCount, NULL);

			std::vector<VkLayerProperties> layerProperties(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());


			bool foundLayer = false;
			for (VkLayerProperties prop : layerProperties) {

				if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0) {
					foundLayer = true;
					break;
				}

			}

			if (!foundLayer) {
				throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
			}
			enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation"); // Alright, we can use this layer.

			uint32_t extensionCount;

			vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, NULL);
			std::vector<VkExtensionProperties> extensionProperties(extensionCount);
			vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, extensionProperties.data());

			bool foundExtension = false;
			for (VkExtensionProperties prop : extensionProperties) {
				if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0) {
					foundExtension = true;
					break;
				}

			}

			if (!foundExtension) {
				throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
			}
			enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
			*/


			/*
			Next, we actually create the instance.

			*/

			/*
			Contains application info. This is actually not that important.
			The only real important field is apiVersion.
			*/
			VkApplicationInfo applicationInfo = {};
			applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			applicationInfo.pApplicationName = "Spirit";
			applicationInfo.applicationVersion = 2.0;
			applicationInfo.pEngineName = "SpiritVulkan";
			applicationInfo.engineVersion = 0.8;
			applicationInfo.apiVersion = VK_API_VERSION_1_1;;

			VkInstanceCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			createInfo.flags = 0;
			createInfo.pApplicationInfo = &applicationInfo;

			// Give our desired layers and extensions to vulkan.
			createInfo.enabledLayerCount = enabledLayers.size();
			createInfo.ppEnabledLayerNames = enabledLayers.data();
			createInfo.enabledExtensionCount = enabledExtensions.size();
			createInfo.ppEnabledExtensionNames = enabledExtensions.data();

			/*
			Actually create the instance.
			Having created the instance, we can actually start using vulkan.
			*/
			VK_CHECK_RESULT(vkCreateInstance(
				&createInfo,
				NULL,
				&instance));

			/*
			Register a callback function for the extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME, so that warnings emitted from the validation
			layer are actually printed.
			*/
			/*

			VkDebugReportCallbackCreateInfoEXT createInfoD = {};
			createInfoD.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
			createInfoD.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
			createInfoD.pfnCallback = &debugReportCallbackFn;

			// We have to explicitly load this function.
			auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
			if (vkCreateDebugReportCallbackEXT == nullptr) {
				throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
			}

			// Create and register callback.
			VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance, &createInfoD, NULL, &debugReportCallback));
			*/

		}

		void findPhysicalDevice() {
			/*
			In this function, we find a physical device that can be used with Vulkan.
			*/

			/*
			So, first we will list all physical devices on the system with vkEnumeratePhysicalDevices .
			*/
			uint32_t deviceCount;
			vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
			if (deviceCount == 0) {
				throw std::runtime_error("could not find a device with vulkan support");
			}

			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

			/*
			Next, we choose a device that can be used for our purposes.

			With VkPhysicalDeviceFeatures(), we can retrieve a fine-grained list of physical features supported by the device.
			However, in this demo, we are simply launching a simple compute shader, and there are no
			special physical features demanded for this task.

			With VkPhysicalDeviceProperties(), we can obtain a list of physical device properties. Most importantly,
			we obtain a list of physical device limitations. For this application, we launch a compute shader,
			and the maximum size of the workgroups and total number of compute shader invocations is limited by the physical device,
			and we should ensure that the limitations named maxComputeWorkGroupCount, maxComputeWorkGroupInvocations and
			maxComputeWorkGroupSize are not exceeded by our application.  Moreover, we are using a storage buffer in the compute shader,
			and we should ensure that it is not larger than the device can handle, by checking the limitation maxStorageBufferRange.

			However, in our application, the workgroup size and total number of shader invocations is relatively small, and the storage buffer is
			not that large, and thus a vast majority of devices will be able to handle it. This can be verified by looking at some devices at_
			http://vulkan.gpuinfo.org/

			Therefore, to keep things simple and clean, we will not perform any such checks here, and just pick the first physical
			device in the list. But in a real and serious application, those limitations should certainly be taken into account.

			*/

			for (VkPhysicalDevice device : devices) {
				if (true) { // As above stated, we do no feature checks, so just accept.
					physicalDevice = device;
					context.physicalDevice = device;
					break;
				}
			}
		}

		// Returns the index of a queue family that supports compute operations. 
		uint32_t getComputeQueueFamilyIndex() {
			uint32_t queueFamilyCount;

			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

			// Retrieve all queue families.
			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

			// Now find a family that supports compute.
			uint32_t i = 0;
			for (; i < queueFamilies.size(); ++i) {
				VkQueueFamilyProperties props = queueFamilies[i];

				if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
					// found a queue with compute. We're done!
					break;
				}
			}

			if (i == queueFamilies.size()) {
				throw std::runtime_error("could not find a queue family that supports operations");
			}

			return i;
		}

		void createDevice() {
			/*
			We create the logical device in this function.
			*/

			/*
			When creating the device, we also specify what queues it has.
			*/
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueFamilyIndex = getComputeQueueFamilyIndex(); // find queue family with compute capability.
			queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
			queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
			float queuePriorities = 1.0;  // we only have one queue, so this is not that imporant. 
			queueCreateInfo.pQueuePriorities = &queuePriorities;

			/*
			Now we create the logical device. The logical device allows us to interact with the physical
			device.
			*/
			VkDeviceCreateInfo deviceCreateInfo = {};

			// Specify any desired device features here. We do not need any for this application, though.
			VkPhysicalDeviceFeatures deviceFeatures = {};

			deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
			deviceCreateInfo.enabledLayerCount = enabledLayers.size();  // need to specify validation layers here as well.
			deviceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
			deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo; // when creating the logical device, we also specify what queues it has.
			deviceCreateInfo.queueCreateInfoCount = 1;
			deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

			VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.
			context.device = device;

			// Get a handle to the only member of the queue family.
			vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
			context.queue = queue;
		}

		// find memory type with desired properties.
		uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
			VkPhysicalDeviceMemoryProperties memoryProperties;

			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

			/*
			How does this search work?
			See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
			*/
			for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
				if ((memoryTypeBits & (1 << i)) &&
					((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
					return i;
			}
			return -1;
		}

		void createBuffer(VkBuffer* buffer, VkDeviceMemory* bufferMemory, uint32_t bufferSize) {
			/*
			We will now create a buffer. We will render the mandelbrot set into this buffer
			in a computer shade later.
			*/

			VkBufferCreateInfo bufferCreateInfo = {};
			bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferCreateInfo.size = bufferSize; // buffer size in bytes. 
			bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // buffer is used as a storage buffer.
			bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

			VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, buffer)); // create buffer.

			/*
			But the buffer doesn't allocate memory for itself, so we must do that manually.
			*/

			/*
			First, we find the memory requirements for the buffer.
			*/
			VkMemoryRequirements memoryRequirements;
			vkGetBufferMemoryRequirements(device, buffer[0], &memoryRequirements);

			/*
			Now use obtained memory requirements info to allocate the memory for the buffer.
			*/
			VkMemoryAllocateInfo allocateInfo = {};
			allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocateInfo.allocationSize = memoryRequirements.size; // specify required memory.
			/*
			There are several types of memory that can be allocated, and we must choose a memory type that:

			1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits).
			2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
			   with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.
			Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily
			visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
			this flag.
			*/
			allocateInfo.memoryTypeIndex = findMemoryType(
				memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

			VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, bufferMemory)); // allocate memory on device.

			// Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory. 
			VK_CHECK_RESULT(vkBindBufferMemory(device, buffer[0], bufferMemory[0], 0));
		}
		void createBufferUBO(VkBuffer* buffer, VkDeviceMemory* bufferMemory, uint32_t bufferSize) {

			VkBufferCreateInfo bufferCreateInfo = {};
			bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferCreateInfo.size = bufferSize; // buffer size in bytes. 
			bufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT; // buffer is used as a storage buffer.
			bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

			VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, buffer)); // create buffer.
			VkMemoryRequirements memoryRequirements;
			vkGetBufferMemoryRequirements(device, buffer[0], &memoryRequirements);
			VkMemoryAllocateInfo allocateInfo = {};
			allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocateInfo.allocationSize = memoryRequirements.size; // specify required memory.
			allocateInfo.memoryTypeIndex = findMemoryType(
				memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

			VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, bufferMemory)); // allocate memory on device.

			VK_CHECK_RESULT(vkBindBufferMemory(device, buffer[0], bufferMemory[0], 0));
		}

		// Read file into array of bytes, and cast to uint32_t*, then return.
		// The data has been padded, so that it fits into an array uint32_t.
		uint32_t* readFile(uint32_t& length, const char* filename) {

			FILE* fp = fopen(filename, "rb");
			if (fp == NULL) {
				printf("Could not find or open file: %s\n", filename);
			}

			// get file size.
			fseek(fp, 0, SEEK_END);
			long filesize = ftell(fp);
			fseek(fp, 0, SEEK_SET);

			long filesizepadded = long(ceil(filesize / 4.0)) * 4;

			// read file contents.
			char* str = new char[filesizepadded];
			fread(str, filesize, sizeof(char), fp);
			fclose(fp);

			// data padding. 
			for (int i = filesize; i < filesizepadded; i++) {
				str[i] = 0;
			}

			length = filesizepadded;
			return (uint32_t*)str;
		}
		void createComputeGradients_noDDI(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 4;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanFFTPlan->context->device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferRegions_Book;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeRegions_Book;
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = bufferRegions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeRegions;
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/all.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanFFTPlan->context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil((SIZES[0]) / float(WORKGROUP_SIZE)), (uint32_t)ceil(SIZES[1] / float(WORKGROUP_SIZE)), (uint32_t)SIZES[2]);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void createConvolution(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan)
		{
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanFFTPlan->context->device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = kernel;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeKernel;
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[1];
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}
			uint32_t dims[3] = { 512,1,1 };
			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/convolution.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				std::array<VkSpecializationMapEntry, 3> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				specializationMapEntries[1].constantID = 2;
				specializationMapEntries[1].size = sizeof(uint32_t);
				specializationMapEntries[1].offset = sizeof(uint32_t);
				specializationMapEntries[2].constantID = 3;
				specializationMapEntries[2].size = sizeof(uint32_t);
				specializationMapEntries[2].offset = 2 * sizeof(uint32_t);

				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = 3 * sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &dims;

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanFFTPlan->context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil((vulkanFFTPlan->axes[1].sampleCount) / ((scalar)dims[0])), (uint32_t)ceil((vulkanFFTPlan->axes[0].sampleCount / 2 + 1) / ((scalar)dims[1])), (uint32_t)ceil(vulkanFFTPlan->axes[2].sampleCount / ((scalar)dims[2])));
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		/*void createConvolutionHermitian(VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 1;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanFFTPlan->context->device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &descriptorPoolConvolutionHermitian);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &descriptorSetLayoutConvolutionHermitian);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = descriptorPoolConvolutionHermitian;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayoutConvolutionHermitian;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, &descriptorSetConvolutionHermitian);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[vulkanFFTPlan->resultInSwapBuffer];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize;
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = descriptorSetConvolutionHermitian;
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/convolutionhermitian.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModuleConvolutionHermitian);
				delete[] code;

				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayoutConvolutionHermitian;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &pipelineLayoutConvolutionHermitian);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = computeShaderModuleConvolutionHermitian;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = pipelineLayoutConvolutionHermitian;


				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanFFTPlan->context->allocator, &pipeline_convolutionhermitian);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBufferConvolutionHermitian)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(commandBufferConvolutionHermitian, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(commandBufferConvolutionHermitian, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayoutConvolutionHermitian, 0, 1, &descriptorSetConvolutionHermitian, 0, NULL);
				vkCmdBindPipeline(commandBufferConvolutionHermitian, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_convolutionhermitian);
				vkCmdDispatch(commandBufferConvolutionHermitian, (uint32_t)ceil((SIZES[0] - 1) / float(WORKGROUP_SIZE)), (uint32_t)ceil((2 * SIZES[1]) / float(WORKGROUP_SIZE)), (uint32_t)(SIZES[2]));
				VK_CHECK_RESULT(vkEndCommandBuffer(commandBufferConvolutionHermitian)); // end recording commands.
			}
		}*/
		void createReadSpins(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanFFTPlan->context->device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = bufferSpinsInit;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/readSpins.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanFFTPlan->context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil((SIZES[0]) / float(WORKGROUP_SIZE)), (uint32_t)ceil(SIZES[1] / float(WORKGROUP_SIZE)), (uint32_t)SIZES[2]);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void createWriteGradient(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanFFTPlan->context->device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradientOut;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeGradient;
					}

					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/writeGradient.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanFFTPlan->context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil((SIZES[0]) / float(WORKGROUP_SIZE)), (uint32_t)ceil(SIZES[1] / float(WORKGROUP_SIZE)), (uint32_t)SIZES[2]);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void createWriteSpins(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanFFTPlan->context->device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferSpinsInit;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}

					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/writeGradient.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanFFTPlan->context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil((SIZES[0]) / float(WORKGROUP_SIZE)), (uint32_t)ceil(SIZES[1] / float(WORKGROUP_SIZE)), (uint32_t)SIZES[2]);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void createZeroPadding(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 4;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanFFTPlan->context->device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[0];
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = bufferRegions_Book;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeRegions_Book;
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = bufferRegions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeRegions;
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/zeropad.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanFFTPlan->context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil((SIZES[0]) / float(WORKGROUP_SIZE)), (uint32_t)ceil(SIZES[1] / float(WORKGROUP_SIZE)), (uint32_t)SIZES[2]);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void createR2C_decomp(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[1];
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/r2c_decomp.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 2 * sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };
				std::array<VkSpecializationMapEntry, 3> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				specializationMapEntries[1].constantID = 2;
				specializationMapEntries[1].size = sizeof(uint32_t);
				specializationMapEntries[1].offset = sizeof(uint32_t);
				specializationMapEntries[2].constantID = 3;
				specializationMapEntries[2].size = sizeof(uint32_t);
				specializationMapEntries[2].offset = 2 * sizeof(uint32_t);
				uint32_t r2c_kernel[3] = { 32,8,1 };
				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = 3 * sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &r2c_kernel;

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}

		}
		void createC2R_decomp(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[1];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[0];
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/c2r_decomp.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 2 * sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				std::array<VkSpecializationMapEntry, 3> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				specializationMapEntries[1].constantID = 2;
				specializationMapEntries[1].size = sizeof(uint32_t);
				specializationMapEntries[1].offset = sizeof(uint32_t);
				specializationMapEntries[2].constantID = 3;
				specializationMapEntries[2].size = sizeof(uint32_t);
				specializationMapEntries[2].offset = 2 * sizeof(uint32_t);

				uint32_t c2r_kernel[3] = { 32,8,1 };
				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = 3 * sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &c2r_kernel;

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}

		}
		void createZeroPaddingRemove(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 4;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanFFTPlan->context->device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = uboDimensions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = sizeof(VulkanDimensions);
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = bufferRegions_Book;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeRegions_Book;
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = bufferRegions;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeRegions;
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/zeropadremove.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanFFTPlan->context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil((SIZES[0]) / float(WORKGROUP_SIZE)), (uint32_t)ceil(SIZES[1] / float(WORKGROUP_SIZE)), (uint32_t)SIZES[2]);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void createFillZero(VulkanCollection* collection, VulkanFFTPlan* vulkanFFTPlan) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanFFTPlan->context->device, &descriptorPoolCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanFFTPlan->context->device, &descriptorSetLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanFFTPlan->context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[0];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanFFTPlan->buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanFFTPlan->bufferSize[0];
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/fillzero.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanFFTPlan->context->device, &pipelineLayoutCreateInfo, vulkanFFTPlan->context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };
				uint32_t zoffset = 3 * vulkanFFTPlan->axes[0].sampleCount * (vulkanFFTPlan->axes[1].sampleCount / 2);
				std::array<VkSpecializationMapEntry, 1> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;

				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &zoffset;

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanFFTPlan->context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanFFTPlan->context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * vulkanFFTPlan->axes[0].sampleCount * (vulkanFFTPlan->axes[1].sampleCount / 2) / 1024.0f), vulkanFFTPlan->axes[2].sampleCount, 1);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void runCommandBuffer(VkCommandBuffer commandBuffer) {
			/*
			Now we shall finally submit the recorded command buffer to a queue.
			*/

			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1; // submit a single command buffer
			submitInfo.pCommandBuffers = &commandBuffer; // the command buffer to submit.
			vkQueueSubmit(context.queue, 1, &submitInfo, context.fence);
			vkWaitForFences(context.device, 1, &context.fence, VK_TRUE, 100000000000);
			vkResetFences(context.device, 1, &context.fence);

			/*
			We submit the command buffer on the queue, at the same time giving a fence.
			*/
			//VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
			/*
			The command will not have finished executing until the fence is signalled.
			So we wait here.
			We will directly after this read our buffer from the GPU,
			and we will not be sure that the command has finished executing unless we wait for the fence.
			Hence, we use a fence here.
			*/
			//VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000));

			//vkDestroyFence(device, fence, NULL);
		}
		void initReduceBuffers() {
			int n = SIZES[0] * SIZES[1] * SIZES[2];
			int localSize = 1024;
			vulkanReduce.bufferNum = 0;
			while (n > 1)
			{
				n = (n + localSize - 1) / localSize;
				vulkanReduce.bufferNum++;
			}
			//vulkanReduce.bufferNum++;
			vulkanReduce.sizes = (int*)malloc(sizeof(int) * vulkanReduce.bufferNum);
			vulkanReduce.buffer = (VkBuffer*)malloc(sizeof(VkBuffer) * vulkanReduce.bufferNum);
			vulkanReduce.deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) * vulkanReduce.bufferNum);
			n = SIZES[0] * SIZES[1] * SIZES[2];
			for (int i = 0; i < vulkanReduce.bufferNum - 1; i++)
			{
				n = (n + localSize - 1) / localSize;
				vulkanReduce.sizes[i] = n;
				createBufferFFT(vulkanReduce.context, &vulkanReduce.buffer[i], &vulkanReduce.deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, 3 * vulkanReduce.sizes[i] * sizeof(float));
			}
			vulkanReduce.sizes[vulkanReduce.bufferNum - 1] = 1;
			createBufferFFT(vulkanReduce.context, &vulkanReduce.buffer[vulkanReduce.bufferNum - 1], &vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 3 * sizeof(float));

		}
		void createReduce(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2 * vulkanReduce.bufferNum;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = vulkanReduce.bufferNum;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout) * vulkanReduce.bufferNum);
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet) * vulkanReduce.bufferNum);
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanReduce.context->device, &descriptorSetLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorSetLayouts[0]);
				for (uint32_t i = 1; i < vulkanReduce.bufferNum; ++i) {
					collection[0].descriptorSetLayouts[i] = collection[0].descriptorSetLayouts[0];
				}

				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = vulkanReduce.bufferNum;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanReduce.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t j = 0; j < vulkanReduce.bufferNum; ++j)
					for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
						VkDescriptorBufferInfo descriptorBufferInfo = { };
						VkWriteDescriptorSet writeDescriptorSet = { };
						if (j == 0) {
							if (i == 0) {
								descriptorBufferInfo.buffer = bufferSpins;
								descriptorBufferInfo.offset = 0;
								descriptorBufferInfo.range = bufferSizeSpins;
							}
							if (i == 1) {
								descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
								descriptorBufferInfo.offset = 0;
								descriptorBufferInfo.range = vulkanReduce.sizes[0] * 3 * sizeof(float);
							}
						}
						else {
							descriptorBufferInfo.buffer = vulkanReduce.buffer[j - 1 + i];
							descriptorBufferInfo.offset = 0;
							descriptorBufferInfo.range = vulkanReduce.sizes[j - 1 + i] * 3 * sizeof(float);
						}
						writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						writeDescriptorSet.dstSet = collection[0].descriptorSets[j];
						writeDescriptorSet.dstBinding = i;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorType = descriptorType[i];
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
						vkUpdateDescriptorSets(vulkanReduce.context->device, 1, &writeDescriptorSet, 0, NULL);
					}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/scan.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = vulkanReduce.bufferNum;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanReduce.context->device, &pipelineLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				// Prepare specialization info block for the shader stage
				struct SpecializationData {
					// Sets the lighting model used in the fragment "uber" shader
					uint32_t local_size_x_id;
					uint32_t sumSubGroupSize;
				} specializationData;
				specializationData.local_size_x_id = 1024;
				specializationData.sumSubGroupSize = 32;
				std::array<VkSpecializationMapEntry, 2> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				specializationMapEntries[1].constantID = 2;
				specializationMapEntries[1].size = sizeof(uint32_t);
				specializationMapEntries[1].offset = sizeof(uint32_t);

				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = 2 * sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &specializationData;
				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					pipelineShaderStageCreateInfo[i].module = collection[0].computeShader;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(vulkanReduce.context->device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, vulkanReduce.context->allocator, collection[0].pipelines);
			}
			int n = SIZES[0] * SIZES[1] * SIZES[2];
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
			vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &n);
			vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(collection[0].commandBuffer, vulkanReduce.sizes[0], 1, 1);
			for (uint32_t i = 0; i < vulkanReduce.bufferNum - 1; ++i) {
				vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &vulkanReduce.sizes[i]);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[i + 1], 0, NULL);
				vkCmdDispatch(collection[0].commandBuffer, vulkanReduce.sizes[i + 1], 1, 1);
			}
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}

		void createReduceDot(VulkanCollection* collection, VkBuffer* input1, VkDeviceSize input1_size, VkBuffer* input2, VkDeviceSize input2_size) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 3;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanReduce.context->device, &descriptorSetLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanReduce.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					VkWriteDescriptorSet writeDescriptorSet = { };
					if (i == 0) {
						//descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						//descriptorBufferInfo.offset = 0;
						//descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
						descriptorBufferInfo.buffer = input1[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = input1_size;
					}
					if (i == 1) {
						//descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						//descriptorBufferInfo.offset = 0;
						//descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
						descriptorBufferInfo.buffer = input2[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = input2_size;
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizes[0] * sizeof(float);
					}
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(vulkanReduce.context->device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/ReduceDot.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 3 * sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanReduce.context->device, &pipelineLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				// Prepare specialization info block for the shader stage
				struct SpecializationData {
					// Sets the lighting model used in the fragment "uber" shader
					uint32_t local_size_x_id;
					uint32_t sumSubGroupSize;
				} specializationData;
				specializationData.local_size_x_id = 1024;
				specializationData.sumSubGroupSize = 32;
				std::array<VkSpecializationMapEntry, 2> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				specializationMapEntries[1].constantID = 2;
				specializationMapEntries[1].size = sizeof(uint32_t);
				specializationMapEntries[1].offset = sizeof(uint32_t);

				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = 2 * sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &specializationData;
				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					pipelineShaderStageCreateInfo[i].module = collection[0].computeShader;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(vulkanReduce.context->device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, vulkanReduce.context->allocator, collection[0].pipelines);
			}
			//uint32_t n[3] = SIZES[0] * SIZES[1] * SIZES[2];
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
			vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(uint32_t), vulkanLBFGS.ReduceDotConsts);
			vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(collection[0].commandBuffer, vulkanReduce.sizes[0], 1, 1);
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}
		void createReduceDotFinish(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2 * (vulkanReduce.bufferNum - 1);
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = (vulkanReduce.bufferNum - 1);
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout) * (vulkanReduce.bufferNum - 1));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet) * (vulkanReduce.bufferNum - 1));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanReduce.context->device, &descriptorSetLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorSetLayouts[0]);
				for (uint32_t i = 1; i < (vulkanReduce.bufferNum - 1); ++i) {
					collection[0].descriptorSetLayouts[i] = collection[0].descriptorSetLayouts[0];
				}

				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = (vulkanReduce.bufferNum - 1);
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanReduce.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t j = 0; j < (vulkanReduce.bufferNum - 1); ++j)
					for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
						VkDescriptorBufferInfo descriptorBufferInfo = { };
						VkWriteDescriptorSet writeDescriptorSet = { };
						descriptorBufferInfo.buffer = vulkanReduce.buffer[j + i];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizes[j + i] * sizeof(float);
						writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						writeDescriptorSet.dstSet = collection[0].descriptorSets[j];
						writeDescriptorSet.dstBinding = i;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorType = descriptorType[i];
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
						vkUpdateDescriptorSets(vulkanReduce.context->device, 1, &writeDescriptorSet, 0, NULL);
					}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/ReduceDotFinish.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = (vulkanReduce.bufferNum - 1);
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanReduce.context->device, &pipelineLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				// Prepare specialization info block for the shader stage
				struct SpecializationData {
					// Sets the lighting model used in the fragment "uber" shader
					uint32_t local_size_x_id;
					uint32_t sumSubGroupSize;
				} specializationData;
				specializationData.local_size_x_id = 1024;
				specializationData.sumSubGroupSize = 32;
				std::array<VkSpecializationMapEntry, 2> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				specializationMapEntries[1].constantID = 2;
				specializationMapEntries[1].size = sizeof(uint32_t);
				specializationMapEntries[1].offset = sizeof(uint32_t);

				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = 2 * sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &specializationData;
				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					pipelineShaderStageCreateInfo[i].module = collection[0].computeShader;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(vulkanReduce.context->device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, vulkanReduce.context->allocator, collection[0].pipelines);
			}
			int n = SIZES[0] * SIZES[1] * SIZES[2];
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
			for (uint32_t i = 0; i < vulkanReduce.bufferNum - 1; ++i) {
				vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &vulkanReduce.sizes[i]);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[i], 0, NULL);
				vkCmdDispatch(collection[0].commandBuffer, vulkanReduce.sizes[i + 1], 1, 1);
			}
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}


		void initLBFGS() {
			vulkanLBFGS.n_lbfgs_memory = 3; // how many previous iterations are stored in the memory
			int n = SIZES[0] * SIZES[1] * SIZES[2];
			/*field<vectorfield> delta_a = field<vectorfield>(vulkanLBFGS.n_lbfgs_memory, vectorfield(n, { 0,0,0 }));
			field<vectorfield> delta_grad = field<vectorfield>(vulkanLBFGS.n_lbfgs_memory, vectorfield(n, { 0,0,0 }));
			scalarfield rho = scalarfield(vulkanLBFGS.n_lbfgs_memory, 0);
			scalarfield alpha = scalarfield(vulkanLBFGS.n_lbfgs_memory, 0);
			vectorfield forces = vectorfield(n, { 0,0,0 });
			vectorfield forces_virtual = vectorfield(n, { 0,0,0 });
			vectorfield searchdir = vectorfield(n, { 0,0,0 });
			vectorfield grad = vectorfield(n, { 0,0,0 });
			vectorfield grad_pr = vectorfield(n, { 0,0,0 });
			vectorfield  q_vec = vectorfield(n, { 0,0,0 });*/

			//vulkanReduce.bufferNum++;
			vulkanLBFGS.bufferSizes = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * (4 + 6));
			vulkanLBFGS.bufferSizes[0] = 3 * vulkanLBFGS.n_lbfgs_memory * n * sizeof(float);
			vulkanLBFGS.bufferSizes[1] = 3 * vulkanLBFGS.n_lbfgs_memory * n * sizeof(float);
			//vulkanLBFGS.bufferSizes[2] = vulkanLBFGS.n_lbfgs_memory * sizeof(float);
			//vulkanLBFGS.bufferSizes[3] = vulkanLBFGS.n_lbfgs_memory * sizeof(float);
			for (int i = 2; i < 4 + 6; i++)
			{
				vulkanLBFGS.bufferSizes[i] = 3 * n * sizeof(float);
			}
			vulkanLBFGS.buffer = (VkBuffer*)malloc(sizeof(VkBuffer) * (4 + 6));
			vulkanLBFGS.deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) * (4 + 6));

			for (int i = 0; i < 2; i++)
			{
				createBufferFFT(vulkanLBFGS.context, &vulkanLBFGS.buffer[i], &vulkanLBFGS.deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanLBFGS.bufferSizes[i]);
			}
			for (int i = 6; i < 10; i++)
			{
				createBufferFFT(vulkanLBFGS.context, &vulkanLBFGS.buffer[i], &vulkanLBFGS.deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanLBFGS.bufferSizes[i]);
			}

		}

		void initVP() {
			int n = SIZES[0] * SIZES[1] * SIZES[2];

			/*vectorfield velocity = vectorfield(n, { 0,0,0 });
			vectorfield searchdir = vectorfield(n, { 0,0,0 });
			vectorfield grad = vectorfield(n, { 0,0,0 });
			vectorfield grad_pr = vectorfield(n, { 0,0,0 });*/

			//vulkanReduce.bufferNum++;
			vulkanLBFGS.bufferSizes = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * (3));

			for (int i = 0; i < 3; i++)
			{
				vulkanLBFGS.bufferSizes[i] = 3 * n * sizeof(float);
			}

			vulkanLBFGS.buffer = (VkBuffer*)malloc(sizeof(VkBuffer) * (3));
			vulkanLBFGS.deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) * (3));

			for (int i = 0; i < 3; i++)
			{
				createBufferFFT(vulkanLBFGS.context, &vulkanLBFGS.buffer[i], &vulkanLBFGS.deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanLBFGS.bufferSizes[i]);
			}

		}

		void createSetDir0(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[7];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[7];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/setDir0.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, collection[0].descriptorSets, 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}

		void createSetDir1(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanReduce.context->device, &descriptorSetLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanReduce.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					VkWriteDescriptorSet writeDescriptorSet = { };
					if (i == 0) {
						//descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						//descriptorBufferInfo.offset = 0;
						//descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					if (i == 1) {
						//descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						//descriptorBufferInfo.offset = 0;
						//descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[9];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[9];
					}
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(vulkanReduce.context->device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/setDir1.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanReduce.context->device, &pipelineLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					pipelineShaderStageCreateInfo[i].module = collection[0].computeShader;
					pipelineShaderStageCreateInfo[i].pName = "main";
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(vulkanReduce.context->device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, vulkanReduce.context->allocator, collection[0].pipelines);
			}
			//uint32_t n[3] = SIZES[0] * SIZES[1] * SIZES[2];
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.
			recordSetDir1(collection);
		}
		void recordSetDir1(VulkanCollection* collection) {

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
			vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &vulkanLBFGS.setDir1Consts);
			vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}

		void createSetdadg(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/setdadg.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, collection[0].descriptorSets, 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] * vulkanLBFGS.n_lbfgs_memory / 1024.0f), 1, 1);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void createApply0(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 5;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[7];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[7];
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[8];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[8];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}

					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/Apply0.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &vulkanLBFGS.apply0Consts);
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, collection[0].descriptorSets, 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}

		void createApply1(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanReduce.context->device, &descriptorSetLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanReduce.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					VkWriteDescriptorSet writeDescriptorSet = { };
					if (i == 0) {
						//descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						//descriptorBufferInfo.offset = 0;
						//descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[9];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[9];
					}
					if (i == 1) {
						//descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						//descriptorBufferInfo.offset = 0;
						//descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
					}
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(vulkanReduce.context->device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/Apply1.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 2 * sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanReduce.context->device, &pipelineLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					pipelineShaderStageCreateInfo[i].module = collection[0].computeShader;
					pipelineShaderStageCreateInfo[i].pName = "main";
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(vulkanReduce.context->device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, vulkanReduce.context->allocator, collection[0].pipelines);
			}
			//uint32_t n[3] = SIZES[0] * SIZES[1] * SIZES[2];
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.
			recordApply1(collection);
		}
		void recordApply1(VulkanCollection* collection) {

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
			vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(uint32_t), &vulkanLBFGS.apply1Consts);
			vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}

		void createApply2(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanReduce.context->device, &descriptorSetLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanReduce.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					VkWriteDescriptorSet writeDescriptorSet = { };
					if (i == 0) {
						//descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						//descriptorBufferInfo.offset = 0;
						//descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					if (i == 1) {
						//descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						//descriptorBufferInfo.offset = 0;
						//descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(vulkanReduce.context->device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/Apply2.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 3 * sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanReduce.context->device, &pipelineLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					pipelineShaderStageCreateInfo[i].module = collection[0].computeShader;
					pipelineShaderStageCreateInfo[i].pName = "main";
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(vulkanReduce.context->device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, vulkanReduce.context->allocator, collection[0].pipelines);
			}
			//uint32_t n[3] = SIZES[0] * SIZES[1] * SIZES[2];
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.
			recordApply2(collection);
		}
		void recordApply2(VulkanCollection* collection) {

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
			vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t), &vulkanLBFGS.apply2Consts);
			vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}

		void createApply3(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 3;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[7];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[7];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[8];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[8];
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/Apply3.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, collection[0].descriptorSets, 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}

		void createScale(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanReduce.context->device, &descriptorSetLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanReduce.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					VkWriteDescriptorSet writeDescriptorSet = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(vulkanReduce.context->device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/scale.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanReduce.context->device, &pipelineLayoutCreateInfo, vulkanReduce.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					pipelineShaderStageCreateInfo[i].module = collection[0].computeShader;
					pipelineShaderStageCreateInfo[i].pName = "main";
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(vulkanReduce.context->device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, vulkanReduce.context->allocator, collection[0].pipelines);
			}
			//uint32_t n[3] = SIZES[0] * SIZES[1] * SIZES[2];
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.
			recordScale(collection);
		}
		void recordScale(VulkanCollection* collection) {

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
			vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(scalar), &vulkanLBFGS.scaling);
			vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}

		void createApplyVP1(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 3;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[2];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[2];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}


					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/ApplyVP1.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(scalar), &vulkanLBFGS.applyVP1Consts);
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, collection[0].descriptorSets, 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void createApplyVP2(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}


					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/ApplyVP2.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				recordApplyVP2(collection);
			}
		}
		void recordApplyVP2(VulkanCollection* collection) {

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
			vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(scalar), &vulkanLBFGS.applyVP2Consts);
			vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}

		void createOsoCalcGradients(VulkanCollection* collection, uint32_t id_grad_buf) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 3;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[id_grad_buf];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[id_grad_buf];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/oso_calc_gradients.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };
				uint32_t pad = SIZES[0] * SIZES[1];
				std::array<VkSpecializationMapEntry, 1> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;

				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize =  sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &pad;



				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;

				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, collection[0].descriptorSets, 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}
		void createOsoRotate(VulkanCollection* collection, uint32_t id_buf_searchdir) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(vulkanLBFGS.context->device, &descriptorPoolCreateInfo, vulkanLBFGS.context->allocator, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[COUNT_OF(descriptorType)];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType[i];
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(vulkanLBFGS.context->device, &descriptorSetLayoutCreateInfo, vulkanLBFGS.context->allocator, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(vulkanLBFGS.context->device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[id_buf_searchdir];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[id_buf_searchdir];
					}

					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType[i];
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				uint32_t filelength;
				uint32_t* code = readFile(filelength, "shaders/oso_rotate.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &collection[0].computeShader);
				delete[] code;
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(vulkanLBFGS.context->device, &pipelineLayoutCreateInfo, vulkanLBFGS.context->allocator, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };
				uint32_t pad = SIZES[0] * SIZES[1];
				std::array<VkSpecializationMapEntry, 1> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;

				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &pad;

				
				
				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				pipelineShaderStageCreateInfo.module = collection[0].computeShader;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(vulkanLBFGS.context->device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, vulkanLBFGS.context->allocator, collection[0].pipelines);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.

				VkCommandBufferBeginInfo beginInfo = {};
				beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
				VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, collection[0].descriptorSets, 0, NULL);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
			}
		}

		void lbfgs_get_searchdir(int& local_iter, scalarfield& rho, scalarfield& alpha,
			const int num_mem, const scalar maxmove) {
			scalar epsilon = sizeof(scalar) == sizeof(float) ? 1e-30 : 1e-300;
			auto timeA = std::chrono::steady_clock::now();
			int nos = SIZES[0] * SIZES[1] * SIZES[2];
			int m_index = local_iter % num_mem; // memory index
			//std::cout << m_index << " m_index\n";
			int c_ind = 0;
			if (local_iter == 0) // gradient descent
			{
				//	bufferTransfer(vulkanLBFGS.context, vulkanLBFGS.buffer[7], bufferGradient, bufferSizeGradient);
				auto time0 = std::chrono::steady_clock::now();
				//bufferTransfer(vulkanLBFGS.context, vulkanLBFGS.buffer[8], vulkanLBFGS.buffer[7], vulkanLBFGS.bufferSizes[7]);
				bufferTransferSolver(vulkanLBFGS.context, &commandBufferTransferSolver);
				auto time1 = std::chrono::steady_clock::now();
				//printf("buffercopy_grad_pr: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() * 0.001);

				//Backend::par::set(grad_pr[img], grad[img], set);
				//auto& dir = searchdir[img];
				//auto& g_cur = grad[img];
				runCommandBuffer(vulkanLBFGS.collectionSetDir0.commandBuffer);
				//Backend::par::set(dir, g_cur, [] SPIRIT_LAMBDA(const Vec & x) { return -x; });
				//auto& da = delta_a[img];
				//auto& dg = delta_grad[img];

				for (int i = 0; i < num_mem; i++)
				{
					rho[i] = 0.0;
					/*auto dai = da[i].data();
					auto dgi = dg[i].data();
					Backend::par::apply(nos, [dai, dgi] SPIRIT_LAMBDA(int idx) {
						dai[idx] = Vec::Zero();
						dgi[idx] = Vec::Zero();
					});*/
				}
				runCommandBuffer(vulkanLBFGS.collectionSetdadg.commandBuffer);
				//auto time2 = std::chrono::steady_clock::now();
				//printf("2nd part init: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() * 0.001);
			}
			else {
				//bufferTransfer(vulkanLBFGS.context, vulkanLBFGS.buffer[7], bufferGradient, bufferSizeGradient);
				//auto da = delta_a[m_index].data();
				//auto dg = delta_grad[m_index].data();
				//auto g = grad.data();
				//auto g_pr = grad_pr.data();
				//auto sd = searchdir.data();
				//vulkanLBFGS.m_index = m_index;
				//vulkanLBFGS.apply0Consts.offset1 = 3 * m_index * nos;
				auto time0 = std::chrono::steady_clock::now();
				runCommandBuffer(vulkanLBFGS.collectionApply0[m_index].commandBuffer);
				/*Backend::par::apply(nos, [da, dg, g, g_pr, sd] SPIRIT_LAMBDA(int idx) {
					da[idx] = sd[idx];
					dg[idx] = g[idx] - g_pr[idx];
				});*/
				auto time1 = std::chrono::steady_clock::now();
				//printf("Apply0: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() * 0.001);
				scalar rinv_temp = 0;
				//rinv_temp += Backend::par::reduce(delta_grad[m_index], delta_a[m_index], dot);
				//vulkanLBFGS.ReduceDotConsts[0] = nos;
				//vulkanLBFGS.ReduceDotConsts[1] = nos * m_index;
				//vulkanLBFGS.ReduceDotConsts[2] = nos * m_index;
				runCommandBuffer(vulkanLBFGS.collectionReduceDot0[m_index].commandBuffer);
				if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
				//if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);

				//writeDataStream(vulkanFFTTransferReduceDot, &rinv_temp);
				auto time2 = std::chrono::steady_clock::now();
				//printf("ReduceDot0: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() * 0.001);
				//void* map;
				//vkMapMemory(vulkanLBFGS.context->device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1], 0, sizeof(scalar), 0, &mapReduce);
				memcpy(&rinv_temp, mapReduce, sizeof(scalar));
				//vkUnmapMemory(device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1]);
				auto time3 = std::chrono::steady_clock::now();
				//printf("CopyDot0: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count() * 0.001);
				//std::cout << rinv_temp << " rinv_temp\n";
				if (rinv_temp > epsilon)
					rho[m_index] = 1.0 / rinv_temp;
				else
				{
					local_iter = 0;
					return lbfgs_get_searchdir(local_iter, rho, alpha, num_mem, maxmove);
				}


				//Backend::par::set(q_vec, grad, set);
				auto time4 = std::chrono::steady_clock::now();
				//bufferTransfer(vulkanLBFGS.context, vulkanLBFGS.buffer[9], vulkanLBFGS.buffer[7], vulkanLBFGS.bufferSizes[7]);
				bufferTransferSolver(vulkanLBFGS.context, &commandBufferTransferSolver2);
				auto time5 = std::chrono::steady_clock::now();
				//printf("copy79: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time5 - time4).count() * 0.001);
				for (int k = num_mem - 1; k > -1; k--)
				{
					c_ind = (k + m_index + 1) % num_mem;
					scalar temp = 0;
					//temp += Backend::par::reduce(delta_a[c_ind], q_vec, dot);
					//vulkanLBFGS.ReduceDotConsts[0] = nos;
					//vulkanLBFGS.ReduceDotConsts[1] = nos* c_ind;
					//vulkanLBFGS.ReduceDotConsts[2] = 0;
					auto time6 = std::chrono::steady_clock::now();
					runCommandBuffer(vulkanLBFGS.collectionReduceDot1[c_ind].commandBuffer);
					if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
					//if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);

					//writeDataStream(vulkanFFTTransferReduceDot, &temp);
					auto time7 = std::chrono::steady_clock::now();
					//printf("ReduceDot1: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time7 - time6).count() * 0.001);
					//vkMapMemory(vulkanLBFGS.context->device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1], 0, sizeof(scalar), 0, &mapReduce);
					memcpy(&temp, mapReduce, sizeof(scalar));
					//vkUnmapMemory(device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1]);
					auto time8 = std::chrono::steady_clock::now();
					//printf("CopyDot1: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time8 - time7).count() * 0.001);
					//std::cout << temp << " temp\n";

					alpha[c_ind] = rho[c_ind] * temp;
					//auto q = q_vec.data();
					//auto a = alpha.data();
					//auto d = delta_grad.data();
					vulkanLBFGS.apply1Consts.offset1 = 3 * c_ind * nos;
					vulkanLBFGS.apply1Consts.alpha = alpha[c_ind];
					auto time9 = std::chrono::steady_clock::now();
					recordApply1(&vulkanLBFGS.collectionApply1);
					runCommandBuffer(vulkanLBFGS.collectionApply1.commandBuffer);
					auto time10 = std::chrono::steady_clock::now();
					//printf("Apply1: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() * 0.001);
					//Backend::par::apply(nos, [c_ind, q, a, d] SPIRIT_LAMBDA(int idx) {
					//	q[idx] += -a[c_ind] * d[c_ind][idx];
					//});
				}

				scalar dy2 = 0;
				//dy2 += Backend::par::reduce(delta_grad[m_index], delta_grad[m_index], dot);
				//vulkanLBFGS.ReduceDotConsts[0] = nos;
				//vulkanLBFGS.ReduceDotConsts[1] = nos * m_index;
				//vulkanLBFGS.ReduceDotConsts[2] = nos * m_index;
				auto time11 = std::chrono::steady_clock::now();
				runCommandBuffer(vulkanLBFGS.collectionReduceDot2[m_index].commandBuffer);
				if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
				//if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);

				//vkMapMemory(vulkanLBFGS.context->device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1], 0, sizeof(scalar), 0, &mapReduce);
				memcpy(&dy2, mapReduce, sizeof(scalar));
				//vkUnmapMemory(device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1]);
				auto time12 = std::chrono::steady_clock::now();
				//printf("ReduceDot2: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time12 - time11).count() * 0.001);
				//writeDataStream(vulkanFFTTransferReduceDot, &dy2);
				//std::cout << dy2 << " dy2\n";
				scalar rhody2 = dy2 * rho[m_index];
				scalar inv_rhody2 = 0.0;
				if (rhody2 > epsilon)
					inv_rhody2 = 1.0 / rhody2;
				else
					inv_rhody2 = 1.0 / (epsilon);
				vulkanLBFGS.setDir1Consts.inv_rhody2 = inv_rhody2;
				auto time13 = std::chrono::steady_clock::now();
				recordSetDir1(&vulkanLBFGS.collectionSetDir1);
				runCommandBuffer(vulkanLBFGS.collectionSetDir1.commandBuffer);
				auto time14 = std::chrono::steady_clock::now();
				//printf("SetDir1: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time14 - time13).count() * 0.001);
				//Backend::par::set(searchdir, q_vec, [inv_rhody2] SPIRIT_LAMBDA(const Vec & q) {
				//	return inv_rhody2 * q;
				//});

				for (int k = 0; k < num_mem; k++)
				{
					if (local_iter < num_mem)
						c_ind = k;
					else
						c_ind = (k + m_index + 1) % num_mem;

					scalar rhopdg = 0;
					//rhopdg += Backend::par::reduce(delta_grad[c_ind], searchdir, dot);
					//vulkanLBFGS.ReduceDotConsts[0] = nos;
					//vulkanLBFGS.ReduceDotConsts[1] = nos * m_index;
					//vulkanLBFGS.ReduceDotConsts[2] = 0;
					auto time15 = std::chrono::steady_clock::now();
					runCommandBuffer(vulkanLBFGS.collectionReduceDot3[c_ind].commandBuffer);
					if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
					//if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);

					//writeDataStream(vulkanFFTTransferReduceDot, &rhopdg);
					//vkMapMemory(vulkanLBFGS.context->device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1], 0, sizeof(scalar), 0, &mapReduce);
					memcpy(&rhopdg, mapReduce, sizeof(scalar));
					//vkUnmapMemory(device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1]);
					auto time16 = std::chrono::steady_clock::now();
					//printf("ReduceDot3: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time16 - time15).count() * 0.001);
					rhopdg *= rho[c_ind];
					//std::cout << rhopdg << " rhopdg\n";

					//auto sd = searchdir.data();
					//auto alph = alpha[c_ind];
					//auto da = delta_a[c_ind].data();
					//Backend::par::apply(nos, [sd, alph, da, rhopdg] SPIRIT_LAMBDA(int idx) {
					//	sd[idx] += (alph - rhopdg) * da[idx];
					//});
					vulkanLBFGS.apply2Consts.offset1 = 3 * c_ind * nos;
					vulkanLBFGS.apply2Consts.alpha = alpha[c_ind];
					vulkanLBFGS.apply2Consts.rhopdg = rhopdg;
					auto time17 = std::chrono::steady_clock::now();
					recordApply2(&vulkanLBFGS.collectionApply2);
					runCommandBuffer(vulkanLBFGS.collectionApply2.commandBuffer);
					auto time18 = std::chrono::steady_clock::now();
					//printf("Apply2: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time18 - time17).count() * 0.001);
				}

				//auto g = grad.data();
				//auto g_pr = grad_pr.data();
				//auto sd = searchdir.data();
				//Backend::par::apply(nos, [g, g_pr, sd] SPIRIT_LAMBDA(int idx) {
				//	g_pr[idx] = g[idx];
				//	sd[idx] = -sd[idx];
				//});
				auto time19 = std::chrono::steady_clock::now();
				runCommandBuffer(vulkanLBFGS.collectionApply3.commandBuffer);
				auto time20 = std::chrono::steady_clock::now();
				//printf("Apply0: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time20 - time19).count() * 0.001);
			}
			local_iter++;
			auto timeB = std::chrono::steady_clock::now();
			//printf("All: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(timeB - timeA).count() * 0.001);
		}

		void vp_get_searchdir(scalar m_temp_inv, scalar dt) {
			int nos = SIZES[0] * SIZES[1] * SIZES[2];
			auto timeA = std::chrono::steady_clock::now();
			runCommandBuffer(vulkanLBFGS.collectionApply1.commandBuffer);

			scalar projection = 0;
			runCommandBuffer(vulkanLBFGS.collectionReduceDot0[0].commandBuffer);
			if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
			//if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);

			memcpy(&projection, mapReduce, sizeof(scalar));

			scalar force_norm = 1;
			runCommandBuffer(vulkanLBFGS.collectionReduceDot1[0].commandBuffer);
			if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
			//if (nos > 1024)	runCommandBuffer(vulkanLBFGS.collectionReduceDotFinish.commandBuffer);

			memcpy(&force_norm, mapReduce, sizeof(scalar));

			scalar ratio = projection / force_norm;
			if (projection <= 0) {
				vulkanLBFGS.applyVP2Consts.grad_mult = -dt * 0.5 * m_temp_inv;
			}
			else {
				vulkanLBFGS.applyVP2Consts.grad_mult = -dt * (ratio + 0.5 * m_temp_inv);
			}

			recordApplyVP2(&vulkanLBFGS.collectionApply2);
			runCommandBuffer(vulkanLBFGS.collectionApply2.commandBuffer);

			auto time0 = std::chrono::steady_clock::now();
			//bufferTransfer(vulkanLBFGS.context, vulkanLBFGS.buffer[2], vulkanLBFGS.buffer[1], vulkanLBFGS.bufferSizes[1]);
			bufferTransferSolver(vulkanLBFGS.context, &commandBufferTransferSolver);
			auto time1 = std::chrono::steady_clock::now();

		}
		void createLBFGS(VulkanLBFGS* vulkanLBFGS) {
			uint32_t nos = SIZES[0] * SIZES[1] * SIZES[2];
			createSetDir0(&vulkanLBFGS->collectionSetDir0);
			createSetDir1(&vulkanLBFGS->collectionSetDir1);
			createSetdadg(&vulkanLBFGS->collectionSetdadg);
			createApply1(&vulkanLBFGS->collectionApply1);
			createApply2(&vulkanLBFGS->collectionApply2);
			createApply3(&vulkanLBFGS->collectionApply3);
			//createReduce(&vulkanLBFGS->collectionReduce);
			vulkanLBFGS->collectionReduceDot0 = (VulkanCollection*)malloc(sizeof(VulkanCollection) * vulkanLBFGS->n_lbfgs_memory);
			vulkanLBFGS->collectionReduceDot1 = (VulkanCollection*)malloc(sizeof(VulkanCollection) * vulkanLBFGS->n_lbfgs_memory);
			vulkanLBFGS->collectionReduceDot2 = (VulkanCollection*)malloc(sizeof(VulkanCollection) * vulkanLBFGS->n_lbfgs_memory);
			vulkanLBFGS->collectionReduceDot3 = (VulkanCollection*)malloc(sizeof(VulkanCollection) * vulkanLBFGS->n_lbfgs_memory);
			vulkanLBFGS->collectionApply0 = (VulkanCollection*)malloc(sizeof(VulkanCollection) * vulkanLBFGS->n_lbfgs_memory);
			for (int i = 0; i < vulkanLBFGS->n_lbfgs_memory; i++) {
				vulkanLBFGS->ReduceDotConsts[0] = nos;
				vulkanLBFGS->ReduceDotConsts[1] = nos * i;
				vulkanLBFGS->ReduceDotConsts[2] = nos * i;
				vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
				createReduceDot(&vulkanLBFGS->collectionReduceDot0[i], &vulkanLBFGS->buffer[1], vulkanLBFGS->bufferSizes[1], &vulkanLBFGS->buffer[0], vulkanLBFGS->bufferSizes[0]);
				vulkanLBFGS->ReduceDotConsts[0] = nos;
				vulkanLBFGS->ReduceDotConsts[1] = nos * i;
				vulkanLBFGS->ReduceDotConsts[2] = 0;
				vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
				createReduceDot(&vulkanLBFGS->collectionReduceDot1[i], &vulkanLBFGS->buffer[0], vulkanLBFGS->bufferSizes[0], &vulkanLBFGS->buffer[9], vulkanLBFGS->bufferSizes[9]);
				vulkanLBFGS->ReduceDotConsts[0] = nos;
				vulkanLBFGS->ReduceDotConsts[1] = nos * i;
				vulkanLBFGS->ReduceDotConsts[2] = nos * i;
				vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
				createReduceDot(&vulkanLBFGS->collectionReduceDot2[i], &vulkanLBFGS->buffer[1], vulkanLBFGS->bufferSizes[1], &vulkanLBFGS->buffer[1], vulkanLBFGS->bufferSizes[1]);
				vulkanLBFGS->ReduceDotConsts[0] = nos;
				vulkanLBFGS->ReduceDotConsts[1] = nos * i;
				vulkanLBFGS->ReduceDotConsts[2] = 0;
				vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
				createReduceDot(&vulkanLBFGS->collectionReduceDot3[i], &vulkanLBFGS->buffer[1], vulkanLBFGS->bufferSizes[1], &vulkanLBFGS->buffer[6], vulkanLBFGS->bufferSizes[6]);
				vulkanLBFGS->apply0Consts.offset1 = 3 * i * nos;
				createApply0(&vulkanLBFGS->collectionApply0[i]);

			}
			vulkanLBFGS->ReduceDotConsts[0] = nos;
			vulkanLBFGS->ReduceDotConsts[1] = 0;
			vulkanLBFGS->ReduceDotConsts[2] = 0;
			vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
			createReduceDot(&collectionReduceDotEnergy, &bufferGradient, bufferSizeGradient, &bufferSpins, bufferSizeSpins);

			vulkanLBFGS->ReduceDotConsts[0] = nos;
			vulkanLBFGS->ReduceDotConsts[1] = 0;
			vulkanLBFGS->ReduceDotConsts[2] = 0;
			vulkanLBFGS->ReduceDotConsts[3] = SIZES[0]*SIZES[1];
			createReduceDot(&vulkanLBFGS->collectionReduceDotScaling, &vulkanLBFGS->buffer[6], vulkanLBFGS->bufferSizes[6], &vulkanLBFGS->buffer[6], vulkanLBFGS->bufferSizes[6]);
			createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish);
			createOsoCalcGradients(&vulkanLBFGS->collectionOsoCalcGradients, 7);
			createOsoRotate(&vulkanLBFGS->collectionOsoRotate, 6);
			createScale(&vulkanLBFGS->collectionScale);
		};
		void createVP(VulkanLBFGS* vulkanLBFGS) {
			uint32_t nos = SIZES[0] * SIZES[1] * SIZES[2];

			vulkanLBFGS->applyVP1Consts.m_temp_inv = 0;
			createApplyVP1(&vulkanLBFGS->collectionApply1);

			vulkanLBFGS->applyVP2Consts.grad_mult = 0;
			createApplyVP2(&vulkanLBFGS->collectionApply2);


			vulkanLBFGS->collectionReduceDot0 = (VulkanCollection*)malloc(sizeof(VulkanCollection));
			vulkanLBFGS->collectionReduceDot1 = (VulkanCollection*)malloc(sizeof(VulkanCollection));

			vulkanLBFGS->ReduceDotConsts[0] = nos;
			vulkanLBFGS->ReduceDotConsts[1] = 0;
			vulkanLBFGS->ReduceDotConsts[2] = 0;
			vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
			createReduceDot(&vulkanLBFGS->collectionReduceDot0[0], &vulkanLBFGS->buffer[0], vulkanLBFGS->bufferSizes[0], &vulkanLBFGS->buffer[1], vulkanLBFGS->bufferSizes[1]);
			vulkanLBFGS->ReduceDotConsts[0] = nos;
			vulkanLBFGS->ReduceDotConsts[1] = 0;
			vulkanLBFGS->ReduceDotConsts[2] = 0;
			vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
			createReduceDot(&vulkanLBFGS->collectionReduceDot1[0], &vulkanLBFGS->buffer[1], vulkanLBFGS->bufferSizes[1], &vulkanLBFGS->buffer[1], vulkanLBFGS->bufferSizes[1]);

			vulkanLBFGS->ReduceDotConsts[0] = nos;
			vulkanLBFGS->ReduceDotConsts[1] = 0;
			vulkanLBFGS->ReduceDotConsts[2] = 0;
			vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
			createReduceDot(&collectionReduceDotEnergy, &bufferGradient, bufferSizeGradient, &bufferSpins, bufferSizeSpins);

			createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish);

			createOsoCalcGradients(&vulkanLBFGS->collectionOsoCalcGradients, 1);
			createOsoRotate(&vulkanLBFGS->collectionOsoRotate, 0);

		};
		void cleanup() {
			/*
			Clean up all Vulkan Resources.
			*/
			vkUnmapMemory(device, vulkanReduce.deviceMemory[vulkanReduce.bufferNum - 1]);

			vkFreeMemory(device, bufferMemorySpins, NULL);
			vkFreeMemory(device, bufferMemoryGradient, NULL);
			vkFreeMemory(device, bufferMemoryRegions_Book, NULL);
			vkFreeMemory(device, bufferMemoryRegions, NULL);
			vkFreeMemory(device, bufferMemoryKernel, NULL);
			vkFreeMemory(device, uboMemoryDimensions, NULL);

			vkDestroyBuffer(device, bufferSpins, NULL);
			vkDestroyBuffer(device, bufferGradient, NULL);
			vkDestroyBuffer(device, bufferRegions_Book, NULL);
			vkDestroyBuffer(device, bufferRegions, NULL);
			vkDestroyBuffer(device, kernel, NULL);
			vkDestroyBuffer(device, uboDimensions, NULL);

			/*vkDestroyShaderModule(device, computeShaderModule, NULL);
			vkDestroyShaderModule(device, computeShaderModuleConvolution, NULL);
			vkDestroyShaderModule(device, computeShaderModuleConvolutionHermitian, NULL);
			vkDestroyShaderModule(device, computeShaderModuleZeroPadding, NULL);
			vkDestroyShaderModule(device, computeShaderModuleZeroPaddingRemove, NULL);
			vkDestroyShaderModule(device, computeShaderModuleFillZero, NULL);

			vkDestroyDescriptorPool(device, descriptorPool, NULL);
			vkDestroyDescriptorPool(device, descriptorPoolConvolution, NULL);
			vkDestroyDescriptorPool(device, descriptorPoolConvolutionHermitian, NULL);
			vkDestroyDescriptorPool(device, descriptorPoolZeroPadding, NULL);
			vkDestroyDescriptorPool(device, descriptorPoolZeroPaddingRemove, NULL);
			vkDestroyDescriptorPool(device, descriptorPoolFillZero, NULL);

			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayoutConvolution, NULL);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayoutConvolutionHermitian, NULL);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayoutZeroPadding, NULL);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayoutZeroPaddingRemove, NULL);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayoutFillZero, NULL);

			vkDestroyPipelineLayout(device, pipelineLayout, NULL);
			vkDestroyPipelineLayout(device, pipelineLayoutConvolution, NULL);
			vkDestroyPipelineLayout(device, pipelineLayoutConvolutionHermitian, NULL);
			vkDestroyPipelineLayout(device, pipelineLayoutZeroPadding, NULL);
			vkDestroyPipelineLayout(device, pipelineLayoutZeroPaddingRemove, NULL);
			vkDestroyPipelineLayout(device, pipelineLayoutFillZero, NULL);

			vkDestroyPipeline(device, pipeline_all, NULL);
			vkDestroyPipeline(device, pipeline_convolution, NULL);
			vkDestroyPipeline(device, pipeline_zeropadding, NULL);
			vkDestroyPipeline(device, pipeline_zeropaddingremove, NULL);
			vkDestroyPipeline(device, pipeline_convolutionhermitian, NULL);
			vkDestroyPipeline(device, pipeline_fillzero, NULL);*/

			vkDestroyCommandPool(device, commandPool, NULL);
			vkDestroyDevice(device, NULL);
			vkDestroyInstance(instance, NULL);
		}
	};
}


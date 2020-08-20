#pragma once
#include <vector>
#include <memory>
#include <string>
#include <chrono>
#include <thread>
#include <iostream>
#include "Spirit_Defines.h"
#include <engine/Vectormath_Defines.hpp>
#include <data/Geometry.hpp>
#include "VulkanInitializers.hpp"
#include <vulkan/vulkan.h>
#include "VkFFT/vkFFT.h"

#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifndef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

#define COUNT_OF(array) (sizeof(array) / sizeof(array[0]))

namespace VulkanCompute
{
	typedef struct {
		bool DDI=true;
		bool DMI=true;
		bool saveGradientComponents=false;
		bool double_precision_rotate=false;
		bool double_precision_gradient = false;
		bool damping = false;
		bool twoD=false;
		bool adaptiveTimeStep = true;
		bool LBFGS_linesearch = false;
		int solver_type=-1;
		int groupedIterations;
		int savePeriod=100;
		int n_lbfgs_memory=3;
		bool performZeropadding[3] = { false, false, false };
		scalar gamma= 0.00176085964411;
		scalar max_move=200;
		int GPU_ID=0;
	} VulkanSpiritLaunchConfiguration;

	typedef struct {
		VkDescriptorPool descriptorPool;
		int descriptorNum=1;
		VkDescriptorSetLayout* descriptorSetLayouts;
		VkDescriptorSet* descriptorSets;
		VkPipelineLayout pipelineLayout;
		VkPipeline* pipelines;
		VkShaderModule computeShader;
		VkCommandBuffer commandBuffer;
	} VulkanCollection;

	typedef struct {
		int bufferNum;
		int bufferNumMax;
		int* sizes;
		int* sizesMax;
		VkDeviceSize bufferSizes;
		VkBuffer* buffer;
		VkBuffer lastMax;
		VkDeviceMemory deviceMemoryLastMax;
		VkDeviceMemory* deviceMemory;
	} VulkanReduce;

	typedef struct {

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
			uint32_t n;
			uint32_t pad;
			scalar m_temp_inv;
		} applyVP1Consts;
		struct ApplyVP2Consts {
			uint32_t pad;
			scalar dt;
			scalar m_temp_inv;
		} applyVP2Consts;
		struct ApplyDepondtConsts {
			scalar dt;
			uint32_t pad;
		} applyDepondtConsts;
		struct ApplyRK4Consts {
			scalar dt;
			uint32_t pad;
		} applyRK4Consts;
		struct SetDir1Consts {
			scalar inv_rhody2;
		} setDir1Consts;
		struct SetApplyLBFGSConsts {
			uint32_t num_mem;
			uint32_t nos;
			uint32_t pad;
			scalar eps;
			uint32_t k;
			scalar max_move;
		}	applyLBFGSConsts;
		VulkanCollection collectionSetDir0;
		VulkanCollection collectionSetDir1;
		VulkanCollection collectionSetdadg;
		VulkanCollection* collectionApply0;
		VulkanCollection collectionApply1;
		VulkanCollection collectionApply2;
		VulkanCollection collectionApply3;
		VulkanCollection collectionApply4;
		VulkanCollection collectionApply5;
		VulkanCollection collectionApply6;
		VulkanCollection collectionApply7;
		VulkanCollection collectionApply8;
		VulkanCollection collectionApply9;
		VulkanCollection collectionReduceDot0;
		VulkanCollection collectionReduceDot1;
		VulkanCollection* collectionReduceDot2;
		VulkanCollection* collectionReduceDot3;
		VulkanCollection collectionReduceDotScaling;
		VulkanCollection collectionReduceDotFinish;
		VulkanCollection collectionReduceEnergyFinish;
		VulkanCollection collectionReduceDotFinish2;
		VulkanCollection collectionReduceDotFinish3;
		VulkanCollection collectionReduceMaxFinish;
		VulkanCollection collectionOsoCalcGradients;
		VulkanCollection collectionReduce;
		VulkanCollection collectionScale;
		VulkanCollection collectionOsoRotate;
		VkSubmitInfo submitInfo = {};
	} VulkanLBFGS;

	typedef struct {
		uint32_t WIDTH;
		uint32_t HEIGHT;
		uint32_t DEPTH;
		uint32_t n;
	} VulkanDimensions;

	class ComputeApplication {
	private:

		uint32_t SIZES[3];
		int WORKGROUP_SIZE = 32; // Workgroup size in compute shader.
		uint32_t LOCAL_SIZE[3];
		uint32_t LOCAL_SIZE_FFT[3];
		uint32_t num_components;
		void* mapReduce;
		regionbook regions_book_local;

		VkInstance instance = {};
		VkDebugReportCallbackEXT debugReportCallback = {};
		VkPhysicalDevice physicalDevice = {};
		VkPhysicalDeviceProperties physicalDeviceProperties = {};
		VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties = {};
		VkDevice device = {};
		VkDebugUtilsMessengerEXT debugMessenger = {};
		uint32_t queueFamilyIndex = {};
		std::vector<const char*> enabledLayers;
		VkQueue queue = {};
		VkCommandPool commandPool = {};
		VkFence fence = {};

		VkFFTConfiguration forward_configuration;
		VkFFTConfiguration convolution_configuration;
		VkFFTApplication app_convolution;
		VkFFTApplication app_kernel;

		VulkanLBFGS vulkanLBFGS = {};
		VkCommandBuffer commandBufferFFT;
		VkCommandBuffer commandBufferiFFT;
		VkCommandBuffer commandBufferKernel;
		VkCommandBuffer commandBufferCollected[7];
		VkCommandBuffer commandBufferReduce;
		VkCommandBuffer commandBufferTransferSolver;
		VkCommandBuffer commandBufferTransferSolver2;
		VkCommandBuffer commandBufferFullVP;
		VkCommandBuffer commandBufferFullLBFGS;
		VkCommandBuffer commandBufferFullDepondt; 
		VkCommandBuffer commandBufferFullRK4; 

		VkBuffer bufferSpins;
		VkBuffer bufferGradient;
		VkBuffer bufferStagingSpins;
		VkBuffer bufferStagingGradient;
		VkBuffer bufferEnergy;
		VkBuffer bufferSpinsInit;
		VkBuffer bufferGradientOut;
		VkBuffer bufferRegions_Book;
		VkBuffer bufferRegions;
		VkBuffer kernel;
		VkBuffer bufferFFT;
		VkBuffer uboDimensions;
		VkDeviceMemory bufferMemorySpins;
		VkDeviceMemory bufferMemoryGradient;
		VkDeviceMemory bufferMemoryStagingSpins;
		VkDeviceMemory bufferMemoryStagingGradient;
		VkDeviceMemory bufferMemoryEnergy;
		VkDeviceMemory bufferMemorySpinsInit;
		VkDeviceMemory bufferMemoryGradientOut;
		VkDeviceMemory bufferMemoryRegions_Book;
		VkDeviceMemory bufferMemoryRegions;
		VkDeviceMemory bufferMemoryKernel;
		VkDeviceMemory bufferMemoryFFT;
		VkDeviceMemory uboMemoryDimensions;
		VkDeviceSize bufferSizeSpins;
		VkDeviceSize bufferSizeGradient;
		VkDeviceSize bufferSizeStagingSpins;
		VkDeviceSize bufferSizeStagingGradient;
		VkDeviceSize bufferSizeEnergy;
		VkDeviceSize bufferSizeRegions_Book;
		VkDeviceSize bufferSizeRegions;
		VkDeviceSize bufferSizeKernel;
		VkDeviceSize bufferSizeFFT;
		VkDeviceSize uboSizeDimensions;

		VulkanCollection collectionGradients_noDDI_nosave;
		VulkanCollection collectionGradients_noDDI_save;
		VulkanCollection collectionZeroPadding;
		VulkanCollection collectionZeroPaddingRemove;
		VulkanCollection collectionFillZero;
		VulkanCollection collectionReadSpins;
		VulkanCollection collectionWriteGradient;
		VulkanCollection collectionWriteSpins;
		VulkanCollection collectionReduceDotEnergy;
		// DDI FFT

		VulkanReduce vulkanReduce;

		const std::vector<const char*> validationLayers = {
			"VK_LAYER_KHRONOS_validation"
		};
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

		VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
			auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
			if (func != nullptr) {
				return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
			}
			else {
				return VK_ERROR_EXTENSION_NOT_PRESENT;
			}
		}

		void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
			auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
			if (func != nullptr) {
				func(instance, debugMessenger, pAllocator);
			}
		}
		static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
			std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

			return VK_FALSE;
		}
		void setupDebugMessenger() {
			if (!enableValidationLayers) return;

			VkDebugUtilsMessengerCreateInfoEXT createInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
			createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			createInfo.pfnUserCallback = debugCallback;

			if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
				throw std::runtime_error("failed to set up debug messenger");
			}
		}
		std::vector<const char*> getRequiredExtensions() {
			std::vector<const char*> extensions;

			if (enableValidationLayers) {
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}

			return extensions;
		}
		bool checkValidationLayerSupport() {
			uint32_t layerCount;
			vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

			std::vector<VkLayerProperties> availableLayers(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

			for (const char* layerName : validationLayers) {
				bool layerFound = false;

				for (const auto& layerProperties : availableLayers) {
					if (strcmp(layerName, layerProperties.layerName) == 0) {
						layerFound = true;
						break;
					}
				}

				if (!layerFound) {
					return false;
				}
			}

			return true;
		}
		void createInstance() {
			if (enableValidationLayers && !checkValidationLayerSupport()) {
				throw std::runtime_error("validation layers creation failed");
			}

			VkApplicationInfo applicationInfo = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
			applicationInfo.pApplicationName = "Vulkan Spirit";
			applicationInfo.applicationVersion = 1.0;
			applicationInfo.pEngineName = "Vulkan Spirit";
			applicationInfo.engineVersion = 1.0;
			applicationInfo.apiVersion = VK_API_VERSION_1_1;;

			VkInstanceCreateInfo createInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
			createInfo.flags = 0;
			createInfo.pApplicationInfo = &applicationInfo;

			auto extensions = getRequiredExtensions();
			createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
			createInfo.ppEnabledExtensionNames = extensions.data();

			VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
			if (enableValidationLayers) {
				createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();
				debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
				debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
				debugCreateInfo.pfnUserCallback = debugCallback;
				createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
			}
			else {
				createInfo.enabledLayerCount = 0;

				createInfo.pNext = nullptr;
			}

			if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
				throw std::runtime_error("instance creation failed");
			}


		}
		void findPhysicalDevice() {

			uint32_t deviceCount;
			vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
			if (deviceCount == 0) {
				throw std::runtime_error("device with vulkan support not found");
			}

			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
			physicalDevice = devices[launchConfiguration.GPU_ID];
						
		}
		uint32_t getComputeQueueFamilyIndex() {
			uint32_t queueFamilyCount;

			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

			uint32_t i = 0;
			for (; i < queueFamilies.size(); ++i) {
				VkQueueFamilyProperties props = queueFamilies[i];

				if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
					break;
				}
			}

			if (i == queueFamilies.size()) {
				throw std::runtime_error("queue family creation failed");
			}

			return i;
		}
		void createDevice() {

			VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
			queueFamilyIndex = getComputeQueueFamilyIndex();
			queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
			queueCreateInfo.queueCount = 1;
			float queuePriorities = 1.0;
			queueCreateInfo.pQueuePriorities = &queuePriorities;
			VkDeviceCreateInfo deviceCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
			VkPhysicalDeviceFeatures deviceFeatures = {};
			deviceFeatures.shaderFloat64 = true;
			deviceCreateInfo.enabledLayerCount = enabledLayers.size();
			deviceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
			deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
			deviceCreateInfo.queueCreateInfoCount = 1;
			deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
			vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device);
			vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

		}
		uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
			VkPhysicalDeviceMemoryProperties memoryProperties = {};

			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

			for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
				if ((memoryTypeBits & (1 << i)) &&
					((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
					return i;
			}
			return -1;
		}

		uint32_t* readShader(uint32_t& length, const char* filename) {

			FILE* fp = fopen(filename, "rb");
			if (fp == NULL) {
				printf("Could not find or open file: %s\n", filename);
			}

			// get file size.
			fseek(fp, 0, SEEK_END);
			long filesize = ftell(fp);
			fseek(fp, 0, SEEK_SET);

			long filesizepadded = long(ceil(filesize / 4.0)) * 4;

			char* str = new char[filesizepadded];
			fread(str, filesize, sizeof(char), fp);
			fclose(fp);

			for (long i = filesize; i < filesizepadded; i++) {
				str[i] = 0;
			}

			length = filesizepadded;
			return (uint32_t*)str;
		}
		void allocateBuffer(VkBuffer* buffer, VkDeviceMemory* deviceMemory, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags propertyFlags, VkDeviceSize size) {
			uint32_t queueFamilyIndices;
			VkBufferCreateInfo bufferCreateInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
			bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			bufferCreateInfo.queueFamilyIndexCount = 1;
			bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndices;
			bufferCreateInfo.size = size;
			bufferCreateInfo.usage = usageFlags;
			vkCreateBuffer(device, &bufferCreateInfo, NULL, buffer);
			VkMemoryRequirements memoryRequirements = {};
			vkGetBufferMemoryRequirements(device, buffer[0], &memoryRequirements);
			VkMemoryAllocateInfo memoryAllocateInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
			memoryAllocateInfo.allocationSize = memoryRequirements.size;
			memoryAllocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, propertyFlags);
			vkAllocateMemory(device, &memoryAllocateInfo, NULL, deviceMemory);
			vkBindBufferMemory(device, buffer[0], deviceMemory[0], 0);
		}
		void transferDataFromCPU(void* arr, VkDeviceSize bufferSize, VkBuffer* buffer) {
			VkDeviceSize stagingBufferSize = bufferSize;
			VkBuffer stagingBuffer = {};
			VkDeviceMemory stagingBufferMemory = {};
			allocateBuffer(&stagingBuffer, &stagingBufferMemory, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBufferSize);

			void* data;
			vkMapMemory(device, stagingBufferMemory, 0, stagingBufferSize, 0, &data);
			memcpy(data, arr, stagingBufferSize);
			vkUnmapMemory(device, stagingBufferMemory);
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VkCommandBuffer commandBuffer = {};
			vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
			VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
			VkBufferCopy copyRegion = {};
			copyRegion.srcOffset = 0;
			copyRegion.dstOffset = 0;
			copyRegion.size = stagingBufferSize;
			vkCmdCopyBuffer(commandBuffer, stagingBuffer, buffer[0], 1, &copyRegion);
			vkEndCommandBuffer(commandBuffer);
			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(queue, 1, &submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
			vkResetFences(device, 1, &fence);
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
			vkDestroyBuffer(device, stagingBuffer, NULL);
			vkFreeMemory(device, stagingBufferMemory, NULL);
		}
		void transferDataToCPU(VkDeviceSize bufferSize, VkBuffer* buffer, VkBuffer* stagingBuffer) {
			VkDeviceSize stagingBufferSize = bufferSize;

			VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VkCommandBuffer commandBuffer = {};
			vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
			VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
			VkBufferCopy copyRegion = {};
			copyRegion.srcOffset = 0;
			copyRegion.dstOffset = 0;
			copyRegion.size = stagingBufferSize;
			vkCmdCopyBuffer(commandBuffer, buffer[0], stagingBuffer[0], 1, &copyRegion);
			vkEndCommandBuffer(commandBuffer);
			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(queue, 1, &submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
			vkResetFences(device, 1, &fence);
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);			
		}
	public:
		VulkanSpiritLaunchConfiguration launchConfiguration;

		void init(regionbook regions_book, intfield regions,  int region_num, std::shared_ptr<Data::Geometry> geometry, VulkanSpiritLaunchConfiguration* conf) {

			launchConfiguration = conf[0];

			//Sample Vulkan project GPU initialization.
			createInstance();
			setupDebugMessenger();
			findPhysicalDevice();
			createDevice();

			VkFenceCreateInfo fenceCreateInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
			fenceCreateInfo.flags = 0;
			vkCreateFence(device, &fenceCreateInfo, NULL, &fence);
			VkCommandPoolCreateInfo commandPoolCreateInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
			commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
			vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool);
			vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);

			SIZES[0] = geometry->n_cells[0];
			SIZES[1] = geometry->n_cells[1];
			SIZES[2] = geometry->n_cells[2];

			regions_book_local = regions_book;
			// Buffer size of the storage buffer that will contain the rendered mandelbrot set.
			bufferSizeSpins = 3 * sizeof(scalar) * SIZES[0] * SIZES[1] * SIZES[2];
			bufferSizeGradient = 3 * sizeof(scalar) * (SIZES[0]) * SIZES[1] * SIZES[2];//we perform FFT directly here
			bufferSizeEnergy = 2*sizeof(scalar) * SIZES[0] * SIZES[1] * SIZES[2];
			bufferSizeRegions_Book = sizeof(Regionvalues) * region_num;
			bufferSizeRegions = sizeof(int) * SIZES[0] * SIZES[1] * SIZES[2];
			uboSizeDimensions = sizeof(VulkanDimensions);
			
			allocateBuffer(&bufferSpins, &bufferMemorySpins, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeSpins);
			allocateBuffer(&bufferSpinsInit, &bufferMemorySpinsInit, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeSpins);
			allocateBuffer(&bufferEnergy, &bufferMemoryEnergy, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeEnergy);
			num_components=2;
			if (launchConfiguration.saveGradientComponents == true) {
				if (launchConfiguration.DDI == true)
					num_components = 7;
				else
					num_components = 6;
			}
			else {
				num_components = 2;
			}
			allocateBuffer(&bufferGradient, &bufferMemoryGradient, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, num_components*bufferSizeGradient);
			allocateBuffer(&bufferGradientOut, &bufferMemoryGradientOut, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, num_components * bufferSizeGradient);
			allocateBuffer(&bufferRegions_Book, &bufferMemoryRegions_Book, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeRegions_Book);
			allocateBuffer(&bufferRegions, &bufferMemoryRegions, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeRegions);
			allocateBuffer(&uboDimensions, &uboMemoryDimensions, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, uboSizeDimensions);

			bufferSizeStagingSpins = bufferSizeSpins;
			bufferSizeStagingGradient= num_components * bufferSizeGradient;

			allocateBuffer(&bufferStagingSpins, &bufferMemoryStagingSpins, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, bufferSizeStagingSpins);
			allocateBuffer(&bufferStagingGradient, &bufferMemoryStagingGradient, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, bufferSizeStagingGradient);

			initReduceBuffers(&vulkanReduce);

			createComputeGradients_noDDI(&collectionGradients_noDDI_nosave, 0);
			createComputeGradients_noDDI(&collectionGradients_noDDI_save, 1);
			createReadSpins(&collectionReadSpins);
			createWriteGradient(&collectionWriteGradient);
			createWriteSpins(&collectionWriteSpins);

			if (launchConfiguration.DDI == true) {
				if (launchConfiguration.performZeropadding[0])
					forward_configuration.size[0] = 2 * SIZES[0];
				else
					forward_configuration.size[0] = SIZES[0];

				if (launchConfiguration.performZeropadding[1])
					forward_configuration.size[1] = 2 * SIZES[1];
				else
					forward_configuration.size[1] = SIZES[1];

				if (SIZES[2] > 1) {
					launchConfiguration.twoD = false;
					forward_configuration.FFTdim = 3;
					if (launchConfiguration.performZeropadding[2])
						forward_configuration.size[2] = 2 * SIZES[2];
					else
						forward_configuration.size[2] = SIZES[2];
				}
				else {
					launchConfiguration.twoD = true;
					forward_configuration.FFTdim = 2;
					forward_configuration.size[2] = 1;
				}

				forward_configuration.performR2C = true;
				forward_configuration.coordinateFeatures = 6;
				forward_configuration.isInputFormatted = false;
				forward_configuration.isOutputFormatted = false;
				forward_configuration.device = &device;

				sprintf(forward_configuration.shaderPath, SHADER_DIR);
				
				bufferSizeKernel = forward_configuration.coordinateFeatures * 2 * sizeof(scalar) * (forward_configuration.size[0] / 2 + 1) * (forward_configuration.size[1]) * (forward_configuration.size[2]);
				allocateBuffer(&kernel, &bufferMemoryKernel, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeKernel);

				forward_configuration.buffer = &kernel;
				forward_configuration.inputBuffer = &kernel;
				forward_configuration.outputBuffer = &kernel;
				forward_configuration.bufferSize = &bufferSizeKernel;
				forward_configuration.inputBufferSize = &bufferSizeKernel;
				forward_configuration.outputBufferSize = &bufferSizeKernel;

				convolution_configuration = forward_configuration;
				convolution_configuration.performZeropadding[0] = launchConfiguration.performZeropadding[0];
				if (forward_configuration.FFTdim > 1)
					convolution_configuration.performZeropadding[1] = launchConfiguration.performZeropadding[1];
				if (forward_configuration.FFTdim > 2)
					convolution_configuration.performZeropadding[2] = launchConfiguration.performZeropadding[2];
				convolution_configuration.performConvolution = true;
				convolution_configuration.symmetricKernel = true;//Specify if convolution kernel is symmetric. In this case we only pass upper triangle part of it in the form of: (xx, xy, yy) for 2d and (xx, xy, xz, yy, yz, zz) for 3d.
				convolution_configuration.matrixConvolution = 3;
				convolution_configuration.coordinateFeatures = 3;
				convolution_configuration.isInputFormatted = true;
				convolution_configuration.isOutputFormatted = true;
				convolution_configuration.kernel = &kernel;
				convolution_configuration.kernelSize = &bufferSizeKernel;

				bufferSizeFFT = convolution_configuration.coordinateFeatures * 2 * sizeof(scalar) * (forward_configuration.size[0] / 2 + 1) * (forward_configuration.size[1]) * (forward_configuration.size[2]);
				allocateBuffer(&bufferFFT, &bufferMemoryFFT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, bufferSizeFFT);

				convolution_configuration.buffer = &bufferFFT;
				convolution_configuration.inputBuffer = &bufferSpins;
				convolution_configuration.outputBuffer = &bufferGradient;
				convolution_configuration.bufferSize = &bufferSizeFFT;
				convolution_configuration.inputBufferSize = &bufferSizeSpins;
				convolution_configuration.outputBufferSize = &bufferSizeGradient;

				app_kernel.initializeVulkanFFT(forward_configuration);
				app_convolution.initializeVulkanFFT(convolution_configuration);
			}

			VulkanDimensions ubo;
			ubo.WIDTH = SIZES[0];
			ubo.HEIGHT = SIZES[1];
			ubo.DEPTH = SIZES[2];
			ubo.n = SIZES[0] * SIZES[1] * SIZES[2];
			transferDataFromCPU(&ubo, sizeof(VulkanDimensions), &uboDimensions);

			updateRegions(regions.data());
			updateRegionsBook(regions_book, region_num);

		}
		void init_solver(int solver_id) {
			launchConfiguration.solver_type = solver_id;
			switch (launchConfiguration.solver_type) {
			case 0:
				initLBFGS();
				createLBFGS(&vulkanLBFGS);
				setIteration0();
				break;
			case 1:
				initVP();
				createVP(&vulkanLBFGS);
				break;
			case 2:
				initDepondt();
				createDepondt(&vulkanLBFGS);
				break;
			case 3:
				initRK4();
				createRK4(&vulkanLBFGS);
				break;
			}
		}
		void updateRegionsBook(regionbook regions_book, int region_num) {
			Regionvalues* data_regions_book = (Regionvalues*)malloc(sizeof(Regionvalues) * region_num);
			for (uint32_t i = 0; i < region_num; ++i) {
				data_regions_book[i] = regions_book[i];
			}
			transferDataFromCPU(data_regions_book, bufferSizeRegions_Book, &bufferRegions_Book);
			delete[] data_regions_book;
		}
		void updateRegions(int* regions) {
			int* data_regions = (int*)malloc(sizeof(int) * SIZES[0] * SIZES[1] * SIZES[2]);
			for (uint32_t i = 0; i < SIZES[0] * SIZES[1] * SIZES[2]; ++i) {
				data_regions[i] = regions[i];
			}
			transferDataFromCPU(data_regions, bufferSizeRegions, &bufferRegions);
			delete[] data_regions;

		}
		void freeLastSolver() {
			switch (launchConfiguration.solver_type) {
			case 0:
				deleteLBFGS();
				deleteCollectionLBFGS(&vulkanLBFGS);
				launchConfiguration.solver_type = -1;
				break;
			case 1:
				deleteCollectionVP(&vulkanLBFGS);
				launchConfiguration.solver_type = -1;
				break;
			
			case 2:
				deleteCollectionDepondt(&vulkanLBFGS);
				launchConfiguration.solver_type = -1;
				break;
			case 3:
				deleteCollectionRK4(&vulkanLBFGS);
				launchConfiguration.solver_type = -1;
				break;
			}
		}
		void transformKernel(double * fft_dipole_inputs) {
			//we transform kernel and store it in GPU. we can upload it in two passes in spins buffer and do r2c fft
			
			scalar* data = (scalar*)malloc(bufferSizeKernel);
			for (uint32_t v = 0; v < forward_configuration.coordinateFeatures; ++v) {
				for (uint32_t k = 0; k < (forward_configuration.size[2]); ++k) {
					for (uint32_t j = 0; j < (forward_configuration.size[1]); ++j) {
						for (uint32_t i = 0; i < (forward_configuration.size[0]); ++i) {
							data[i + j * (forward_configuration.size[0]) + k * (forward_configuration.size[0]+2) * (forward_configuration.size[1]) + v * (forward_configuration.size[0]+2) * (forward_configuration.size[1]) * (forward_configuration.size[2])] = (scalar)fft_dipole_inputs[i + j * forward_configuration.size[0] + k * (forward_configuration.size[0]) * (forward_configuration.size[1]) + v * (forward_configuration.size[0]) * (forward_configuration.size[1]) * (forward_configuration.size[2])];
						}
					}
				}
			}
			
			transferDataFromCPU(data, bufferSizeKernel, &kernel);
			delete[] data;

			VkCommandBufferAllocateInfo commandBufferAllocateInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			VkCommandBuffer commandBuffer = {};
			vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);
			VkCommandBufferBeginInfo commandBufferBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
			app_kernel.VkFFTAppend(commandBuffer);
			vkEndCommandBuffer(commandBuffer);
			VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(queue, 1, &submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
			vkResetFences(device, 1, &fence);
			vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
		}

		void readInitialSpins(void* in) {
			transferDataFromCPU(in, bufferSizeSpins, &bufferSpinsInit);
			runCommandBuffer(&collectionReadSpins.commandBuffer);
		}
		void setIteration0() {

			uint32_t* iter = (uint32_t*)malloc(vulkanLBFGS.bufferSizes[6]);
			for (uint32_t i = 0; i < vulkanLBFGS.bufferSizes[6] / sizeof(scalar); ++i) {
				iter[i] = 0;
			}
			transferDataFromCPU(iter, vulkanLBFGS.bufferSizes[6], &vulkanLBFGS.buffer[6]);
		
		}

		void copySpinsToCPU(scalar* spins, bool* allow_copy) {
			void* data;
			vkMapMemory(device, bufferMemoryStagingSpins, 0, bufferSizeStagingSpins, 0, &data);
			memcpy(spins, data, bufferSizeStagingSpins);
			vkUnmapMemory(device, bufferMemoryStagingSpins);

			allow_copy[0] = true;
		}
		void copyGradientToCPU(scalar* gradient_contributions_per_spin, bool* allow_copy) {
			void* data;
			vkMapMemory(device, bufferMemoryStagingGradient, 0, bufferSizeStagingGradient, 0, &data);
			memcpy(gradient_contributions_per_spin, data, bufferSizeStagingGradient);
			vkUnmapMemory(device, bufferMemoryStagingGradient);
			
			allow_copy[0] = true;
		}
		void writeSpins(scalar* spins, bool* allow_copy) {
			runCommandBuffer(&collectionWriteSpins.commandBuffer);
			transferDataToCPU(bufferSizeSpins, &bufferSpinsInit, &bufferStagingSpins);
			std::thread t1(&ComputeApplication::copySpinsToCPU, this, spins, allow_copy);
			t1.detach();
			

		}
		void writeGradient(scalar* gradient_contributions_per_spin, bool* allow_copy) {
			runCommandBuffer(&collectionWriteGradient.commandBuffer);
			transferDataToCPU(bufferSizeStagingGradient, &bufferGradientOut, &bufferStagingGradient);
			std::thread t1(&ComputeApplication::copyGradientToCPU, this, gradient_contributions_per_spin, allow_copy);
			t1.detach();
		}
		void bufferTransferSolver(VkCommandBuffer* commandBuffer) {

			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = commandBuffer;
			vkQueueSubmit(queue, 1, &submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
			vkResetFences(device, 1, &fence);
		}

		void getEnergy(scalar* energy, Vector3* meanMag, scalar* MaxForce, scalar* time) {
			scalar temp[18];

			void* map;
			vkMapMemory(device, vulkanReduce.deviceMemoryLastMax, 0, 18 * sizeof(scalar), 0, &map);

			memcpy(&temp, map, 18 * sizeof(scalar));

			vkUnmapMemory(device, vulkanReduce.deviceMemoryLastMax);

			for (int i = 0; i < 5; i++) {
				energy[i] = temp[i] * 0.5f * regions_book_local[0].Ms;
			}
			meanMag[0][0] = temp[5];
			meanMag[0][1] = temp[6];
			meanMag[0][2] = temp[7];
			MaxForce[0]= temp[16];
			time[0] = temp[17];
			return;
		}
		scalar getMaxForce() {
			scalar MaxForce = 0;
			void* map;
			vkMapMemory(device, vulkanReduce.deviceMemoryLastMax, 8 * sizeof(scalar), sizeof(float), 0, &map);
			memcpy(&MaxForce, map, sizeof(float));
			vkUnmapMemory(device, vulkanReduce.deviceMemoryLastMax);
			return MaxForce;
		}
		
		void createComputeGradients_noDDI(VulkanCollection* collection, uint32_t save_energy) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 6;
				//collection->descriptorNum = 6;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}
			uint32_t ddi_bool=1;
			uint32_t damping = 1;
			if (launchConfiguration.DDI == true)
				ddi_bool = 1;
			else
				ddi_bool = 0;
			if (launchConfiguration.damping == true)
				damping = 1;
			else
				damping = 0;
			
			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
						descriptorBufferInfo.range = num_components * bufferSizeGradient;
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
					if (i == 5) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = 9 * vulkanReduce.sizesMax[0] * sizeof(float);
					}
					if (i == 6) {
						descriptorBufferInfo.buffer = bufferEnergy;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeEnergy;
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
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };
				
				uint32_t specialization[5] = { 32, num_components, ddi_bool, damping, save_energy};
				std::array<VkSpecializationMapEntry, 5	> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				specializationMapEntries[1].constantID = 2;
				specializationMapEntries[1].size = sizeof(uint32_t);
				specializationMapEntries[1].offset = sizeof(uint32_t);
				specializationMapEntries[2].constantID = 3;
				specializationMapEntries[2].size = sizeof(uint32_t);
				specializationMapEntries[2].offset = 2*sizeof(uint32_t);
				specializationMapEntries[3].constantID = 4;
				specializationMapEntries[3].size = sizeof(uint32_t);
				specializationMapEntries[3].offset = 3 * sizeof(uint32_t);
				specializationMapEntries[4].constantID = 5;
				specializationMapEntries[4].size = sizeof(uint32_t);
				specializationMapEntries[4].offset = 4 * sizeof(uint32_t);
				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = 5*sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &specialization;

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_gradient == true)
					code = readShader(filelength, "shaders/gradient_double.spv");
				else
					code = readShader(filelength, "shaders/gradient_float.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;

				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}
			
		}
		void recordComputeGradients_noDDIAppend(VulkanCollection* collection, VkCommandBuffer* commandBuffer) {
			vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdDispatch(commandBuffer[0], (uint32_t)ceil((SIZES[0]) / float(WORKGROUP_SIZE)), (uint32_t)ceil(SIZES[1] / float(WORKGROUP_SIZE)), (uint32_t)SIZES[2]);

		}

		void createReadSpins(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 4;
				//collection->descriptorNum = 5;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/readSpins.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;

				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
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
		void createWriteGradient(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 2;
				//collection->descriptorNum = 3;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
						descriptorBufferInfo.range = num_components*bufferSizeGradient;
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradientOut;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = num_components*bufferSizeGradient;
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
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				std::array<VkSpecializationMapEntry, 1> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				
				
				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &num_components;

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/writeGradient.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;

				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
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
		void createWriteSpins(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[2] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorPoolSize[0].descriptorCount = 1;
				descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[1].descriptorCount = 2;
				//collection->descriptorNum = 3;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/writeSpins.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
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
	
		void runCommandBuffer(VkCommandBuffer *commandBuffer) {
			VkSubmitInfo submitInfo = {};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1; 
			submitInfo.pCommandBuffers = commandBuffer; 
			vkQueueSubmit(queue, 1, &submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
			vkResetFences(device, 1, &fence);
		}
		void initReduceBuffers(VulkanReduce * vulkanReduce) {
			int n = 3*SIZES[0] * SIZES[1] * SIZES[2];
			int localSize = 1024;
			vulkanReduce->bufferNum = 0;
			while (n > 1)
			{
				n = (n + localSize - 1) / localSize;
				vulkanReduce->bufferNum++;
			}
			n = SIZES[0] * SIZES[1] * SIZES[2];
			vulkanReduce->bufferNumMax = 0;
			while (n > 1)
			{
				n = (n + localSize - 1) / localSize;
				vulkanReduce->bufferNumMax++;
			}

			vulkanReduce->sizes = (int*)malloc(sizeof(int) * vulkanReduce->bufferNum);
			vulkanReduce->sizesMax = (int*)malloc(sizeof(int) * vulkanReduce->bufferNumMax);
			vulkanReduce->buffer = (VkBuffer*)malloc(sizeof(VkBuffer) * vulkanReduce->bufferNum);
			vulkanReduce->deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) * vulkanReduce->bufferNum);
			n = 3*SIZES[0] * SIZES[1] * SIZES[2];
			for (int i = 0; i < vulkanReduce->bufferNum ; i++)
			{
				n = (n + localSize - 1) / localSize;
				vulkanReduce->sizes[i] = n;
				allocateBuffer(&vulkanReduce->buffer[i], &vulkanReduce->deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, 17*vulkanReduce->sizes[i] * sizeof(float));
			}
			
			n = SIZES[0] * SIZES[1] * SIZES[2];
			for (int i = 0; i < vulkanReduce->bufferNumMax; i++)
			{
				n = (n + localSize - 1) / localSize;
				vulkanReduce->sizesMax[i] = n;
			}
			allocateBuffer(&vulkanReduce->lastMax, &vulkanReduce->deviceMemoryLastMax, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 34*sizeof(float));
		}
		void deleteReduceBuffers(VulkanReduce* vulkanReduce) {
			
			for (int i = 0; i < vulkanReduce->bufferNum; i++)
			{
				vkFreeMemory(device, vulkanReduce->deviceMemory[i], NULL);
				vkDestroyBuffer(device, vulkanReduce->buffer[i], NULL);
			}

			vkFreeMemory(device, vulkanReduce->deviceMemoryLastMax, NULL);
			vkDestroyBuffer(device, vulkanReduce->lastMax, NULL);
			
			//vulkanReduce->sizes[vulkanReduce->bufferNum - 1] = 1;
			//createBufferFFT(vulkanReduce->context, &vulkanReduce->buffer[vulkanReduce->bufferNum - 1], &deviceMemory[vulkanReduce->bufferNum - 1], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 3 * sizeof(float));

		}
		void createReduce(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2 * vulkanReduce.bufferNum;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = vulkanReduce.bufferNum;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout) * vulkanReduce.bufferNum);
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet) * vulkanReduce.bufferNum);
				collection->descriptorNum = vulkanReduce.bufferNum;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				for (uint32_t i = 1; i < vulkanReduce.bufferNum; ++i) {
					collection[0].descriptorSetLayouts[i] = collection[0].descriptorSetLayouts[0];
				}

				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = vulkanReduce.bufferNum;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
						vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
					}
			}

			{
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = vulkanReduce.bufferNum;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(uint32_t);
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				struct SpecializationData {
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
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/scan.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
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
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 4 * sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
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
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/ReduceDot.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);

			}
			//uint32_t n[3] = SIZES[0] * SIZES[1] * SIZES[2];
			/*VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
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
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.*/
		}
		void createReduceDotFinish(VulkanCollection* collection, VulkanReduce * vulkanReduce, uint32_t num_reduce) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2 * (vulkanReduce->bufferNum - 1);
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = (vulkanReduce->bufferNum - 1);
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout) * (vulkanReduce->bufferNum - 1));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet) * (vulkanReduce->bufferNum - 1));
				collection->descriptorNum = vulkanReduce->bufferNum - 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				for (uint32_t i = 1; i < (vulkanReduce->bufferNum - 1); ++i) {
					collection[0].descriptorSetLayouts[i] = collection[0].descriptorSetLayouts[0];
				}

				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = (vulkanReduce->bufferNum - 1);
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t j = 0; j < (vulkanReduce->bufferNum - 1); ++j)
					for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
						VkDescriptorBufferInfo descriptorBufferInfo = { };
						VkWriteDescriptorSet writeDescriptorSet = { };
						descriptorBufferInfo.buffer = vulkanReduce->buffer[j + i];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = num_reduce* vulkanReduce->sizes[j + i] * sizeof(float);
						writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						writeDescriptorSet.dstSet = collection[0].descriptorSets[j];
						writeDescriptorSet.dstBinding = i;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorType = descriptorType[i];
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
						vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
					}
			}

			{
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = (vulkanReduce->bufferNum - 1);
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				// Prepare specialization info block for the shader stage
				struct SpecializationData {
					// Sets the lighting model used in the fragment "uber" shader
					uint32_t local_size_x_id;
					uint32_t sumSubGroupSize;
					uint32_t num_reduce;
				} specializationData;
				specializationData.local_size_x_id = 1024;
				specializationData.sumSubGroupSize = 32;
				specializationData.num_reduce = num_reduce;
				std::array<VkSpecializationMapEntry, 3> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;
				specializationMapEntries[1].constantID = 2;
				specializationMapEntries[1].size = sizeof(uint32_t);
				specializationMapEntries[1].offset = sizeof(uint32_t);
				specializationMapEntries[2].constantID = 3;
				specializationMapEntries[2].size = sizeof(uint32_t);
				specializationMapEntries[2].offset = 2* sizeof(uint32_t);
				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = 3 * sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &specializationData;
				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/ReduceDotFinish.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
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
			for (uint32_t i = 0; i < vulkanReduce->bufferNum - 1; ++i) {

				vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &vulkanReduce->sizes[i]);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[i], 0, NULL);
				vkCmdDispatch(collection[0].commandBuffer, vulkanReduce->sizes[i + 1], 1, 1);
			}
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}

		void createReduceEnergy(VulkanCollection* collection, VkBuffer* input1, VkDeviceSize input1_size, VkBuffer* input2, VkDeviceSize input2_size) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 3;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
						descriptorBufferInfo.range = 4*vulkanReduce.sizesMax[0] * sizeof(float);
					}
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
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 4 * sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
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
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/ReduceEnergy.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
			}
			//uint32_t n[3] = SIZES[0] * SIZES[1] * SIZES[2];
			/*VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
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
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.*/
		}
		void createReduceEnergyFinish(VulkanCollection* collection, VulkanReduce* vulkanReduce, uint32_t num_reduce) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2 * (vulkanReduce->bufferNumMax - 1);
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = (vulkanReduce->bufferNumMax - 1);
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout) * (vulkanReduce->bufferNumMax - 1));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet) * (vulkanReduce->bufferNumMax - 1));
				collection->descriptorNum = vulkanReduce->bufferNum - 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				for (uint32_t i = 1; i < (vulkanReduce->bufferNumMax - 1); ++i) {
					collection[0].descriptorSetLayouts[i] = collection[0].descriptorSetLayouts[0];
				}

				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = (vulkanReduce->bufferNumMax - 1);
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t j = 0; j < (vulkanReduce->bufferNumMax - 1); ++j)
					for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
						VkDescriptorBufferInfo descriptorBufferInfo = { };
						VkWriteDescriptorSet writeDescriptorSet = { };
						if (i + j < vulkanReduce->bufferNumMax - 1) {
							descriptorBufferInfo.buffer = vulkanReduce->buffer[j + i];
						}
						else {
							descriptorBufferInfo.buffer = vulkanReduce->lastMax;
						}
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = num_reduce * vulkanReduce->sizesMax[j + i] * sizeof(float);
						writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						writeDescriptorSet.dstSet = collection[0].descriptorSets[j];
						writeDescriptorSet.dstBinding = i;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorType = descriptorType[i];
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
						vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
					}
			}

			{
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = (vulkanReduce->bufferNumMax - 1);
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				// Prepare specialization info block for the shader stage
				struct SpecializationData {
					// Sets the lighting model used in the fragment "uber" shader
					uint32_t local_size_x_id;
					uint32_t sumSubGroupSize;
					uint32_t num_reduce;
				} specializationData;
				specializationData.local_size_x_id = 1024;
				specializationData.sumSubGroupSize = 32;
				specializationData.num_reduce = num_reduce;
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
				specializationInfo.pData = &specializationData;
				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/ReduceDotFinish.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
			}
			int n = SIZES[0] * SIZES[1] * SIZES[2];
			
		}
		void createReduceMaxFinish(VulkanCollection* collection, VulkanReduce* vulkanReduce, uint32_t num_reduce) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2 * (vulkanReduce->bufferNumMax - 1);
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;
				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = (vulkanReduce->bufferNumMax - 1);
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout) * (vulkanReduce->bufferNumMax - 1));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet) * (vulkanReduce->bufferNumMax - 1));
				collection->descriptorNum = vulkanReduce->bufferNumMax - 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				for (uint32_t i = 1; i < (vulkanReduce->bufferNumMax - 1); ++i) {
					collection[0].descriptorSetLayouts[i] = collection[0].descriptorSetLayouts[0];
				}

				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = (vulkanReduce->bufferNumMax - 1);
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t j = 0; j < (vulkanReduce->bufferNumMax - 1); ++j)
					for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {

						VkDescriptorBufferInfo descriptorBufferInfo = { };
						VkWriteDescriptorSet writeDescriptorSet = { };
						if (i + j < vulkanReduce->bufferNumMax - 1) {
							descriptorBufferInfo.buffer = vulkanReduce->buffer[j + i];
							descriptorBufferInfo.offset = 16 * vulkanReduce->sizesMax[j + i] * sizeof(scalar);
						}
						else {
							descriptorBufferInfo.buffer = vulkanReduce->lastMax;
							descriptorBufferInfo.offset = 16*sizeof(scalar);
						}
						
						descriptorBufferInfo.range = num_reduce * vulkanReduce->sizesMax[j + i] * sizeof(float);
						writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						writeDescriptorSet.dstSet = collection[0].descriptorSets[j];
						writeDescriptorSet.dstBinding = i;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorType = descriptorType[i];
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
						vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
					}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = (vulkanReduce->bufferNumMax - 1);
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				// Prepare specialization info block for the shader stage
				struct SpecializationData {
					// Sets the lighting model used in the fragment "uber" shader
					uint32_t local_size_x_id;
					uint32_t sumSubGroupSize;
					uint32_t num_reduce;
				} specializationData;
				specializationData.local_size_x_id = 1024;
				specializationData.sumSubGroupSize = 32;
				specializationData.num_reduce = num_reduce;
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
				specializationInfo.pData = &specializationData;
				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/ReduceMaxFinish.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
			}
			int n = SIZES[0] * SIZES[1] * SIZES[2];
			
		}
		void createReduceVPFinish(VulkanCollection* collection, VulkanReduce* vulkanReduce, uint32_t num_reduce) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2 * (vulkanReduce->bufferNum - 1);
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = (vulkanReduce->bufferNum - 1);
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout) * (vulkanReduce->bufferNum - 1));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet) * (vulkanReduce->bufferNum - 1));
				collection->descriptorNum = vulkanReduce->bufferNum - 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				for (uint32_t i = 1; i < (vulkanReduce->bufferNum - 1); ++i) {
					collection[0].descriptorSetLayouts[i] = collection[0].descriptorSetLayouts[0];
				}

				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = (vulkanReduce->bufferNum - 1);
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t j = 0; j < (vulkanReduce->bufferNum - 1); ++j)
					for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {


						VkDescriptorBufferInfo descriptorBufferInfo = { };
						VkWriteDescriptorSet writeDescriptorSet = { };
						if (i + j < vulkanReduce->bufferNum - 1) {
							descriptorBufferInfo.buffer = vulkanReduce->buffer[j + i];
							descriptorBufferInfo.offset = 0;
						}
						else {
							descriptorBufferInfo.buffer = vulkanReduce->lastMax;
							descriptorBufferInfo.offset = 32 * sizeof(scalar);
						}

						descriptorBufferInfo.range = num_reduce * vulkanReduce->sizes[j + i] * sizeof(float);
						writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						writeDescriptorSet.dstSet = collection[0].descriptorSets[j];
						writeDescriptorSet.dstBinding = i;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorType = descriptorType[i];
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
						vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
					}
			}

			{
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = (vulkanReduce->bufferNum - 1);
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				// Prepare specialization info block for the shader stage
				struct SpecializationData {
					// Sets the lighting model used in the fragment "uber" shader
					uint32_t local_size_x_id;
					uint32_t sumSubGroupSize;
					uint32_t num_reduce;
				} specializationData;
				specializationData.local_size_x_id = 1024;
				specializationData.sumSubGroupSize = 32;
				specializationData.num_reduce = num_reduce;
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
				specializationInfo.pData = &specializationData;
				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/ReduceDotFinish.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
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
			for (uint32_t i = 0; i < vulkanReduce->bufferNum - 1; ++i) {

				vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &vulkanReduce->sizes[i]);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[i], 0, NULL);
				vkCmdDispatch(collection[0].commandBuffer, vulkanReduce->sizes[i + 1], 1, 1);
			}
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}
		void createReduceDotFinishLBFGS(VulkanCollection* collection, VulkanReduce* vulkanReduce, uint32_t num_reduce) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 3 * (vulkanReduce->bufferNum - 1);
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = (vulkanReduce->bufferNum - 1);
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout) * (vulkanReduce->bufferNum - 1));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet) * (vulkanReduce->bufferNum - 1));
				collection->descriptorNum = vulkanReduce->bufferNum - 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);
				for (uint32_t i = 1; i < (vulkanReduce->bufferNum - 1); ++i) {
					collection[0].descriptorSetLayouts[i] = collection[0].descriptorSetLayouts[0];
				}

				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = (vulkanReduce->bufferNum - 1);
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t j = 0; j < (vulkanReduce->bufferNum - 1); ++j)
					for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
						VkDescriptorBufferInfo descriptorBufferInfo = { };
						VkWriteDescriptorSet writeDescriptorSet = { };
						if (i < 2) {
							descriptorBufferInfo.buffer = vulkanReduce->buffer[j + i];
							descriptorBufferInfo.offset = 0;
							descriptorBufferInfo.range = num_reduce * vulkanReduce->sizes[j + i] * sizeof(float);
						}
						if (i == 2) {
							descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
							descriptorBufferInfo.offset = 0;
							descriptorBufferInfo.range = sizeof(uint32_t);
						}
						writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						writeDescriptorSet.dstSet = collection[0].descriptorSets[j];
						writeDescriptorSet.dstBinding = i;
						writeDescriptorSet.dstArrayElement = 0;
						writeDescriptorSet.descriptorType = descriptorType[i];
						writeDescriptorSet.descriptorCount = 1;
						writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
						vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
					}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = (vulkanReduce->bufferNum - 1);
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = sizeof(uint32_t);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				// Prepare specialization info block for the shader stage
				struct SpecializationData {
					// Sets the lighting model used in the fragment "uber" shader
					uint32_t local_size_x_id;
					uint32_t sumSubGroupSize;
					uint32_t num_reduce;
				} specializationData;
				specializationData.local_size_x_id = 1024;
				specializationData.sumSubGroupSize = 32;
				specializationData.num_reduce = num_reduce;
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
				specializationInfo.pData = &specializationData;
				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/ReduceDotFinishLBFGS.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					pipelineShaderStageCreateInfo[i].pSpecializationInfo = &specializationInfo;
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
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
			for (uint32_t i = 0; i < vulkanReduce->bufferNum - 1; ++i) {

				vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &vulkanReduce->sizes[i]);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[i], 0, NULL);
				vkCmdDispatch(collection[0].commandBuffer, vulkanReduce->sizes[i + 1], 1, 1);
			}
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}

		void recordReduceDotFinishAppend(VulkanCollection* collection, VkCommandBuffer* commandBuffer, VulkanReduce* vulkanReduce, VkMemoryBarrier* memory_barrier) {
			for (uint32_t i = 0; i < vulkanReduce->bufferNum - 1; ++i) {

				vkCmdPushConstants(commandBuffer[0], collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &vulkanReduce->sizes[i]);
				vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[i], 0, NULL);
				vkCmdDispatch(commandBuffer[0], vulkanReduce->sizes[i + 1], 1, 1);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, memory_barrier, 0, NULL, 0, NULL);

			}
		}

		void recordReduceMaxFinishAppend(VulkanCollection* collection, VkCommandBuffer* commandBuffer, VulkanReduce* vulkanReduce, VkMemoryBarrier* memory_barrier) {
			for (uint32_t i = 0; i < vulkanReduce->bufferNumMax - 1; ++i) {

				vkCmdPushConstants(commandBuffer[0], collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &vulkanReduce->sizesMax[i]);
				vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[i], 0, NULL);
				vkCmdDispatch(commandBuffer[0], vulkanReduce->sizesMax[i + 1], 1, 1);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, memory_barrier, 0, NULL, 0, NULL);

			}
		}


		
	
		
		void createSetDir0(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/setDir0.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
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
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
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
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/setDir1.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
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
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/setdadg.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
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
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/Apply0.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
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
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
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
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/Apply1.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
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
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
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
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/Apply2.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
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
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/Apply3.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
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
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &collection[0].descriptorSetLayouts[0]);


				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
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
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo[1] = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo[1] = { };

				for (uint32_t i = 0; i < 1; ++i) {
					pipelineShaderStageCreateInfo[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
					pipelineShaderStageCreateInfo[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
					uint32_t filelength;
					uint32_t* code = readShader(filelength, "shaders/scale.spv");
					VkShaderModuleCreateInfo createInfo = {};
					createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
					createInfo.pCode = code;
					createInfo.codeSize = filelength;
					vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo[i].module);
					delete[] code;
					pipelineShaderStageCreateInfo[i].pName = "main";
					computePipelineCreateInfo[i].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
					computePipelineCreateInfo[i].stage = pipelineShaderStageCreateInfo[i];
					computePipelineCreateInfo[i].layout = collection[0].pipelineLayout;
				}

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo[0].module, NULL);
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

		void createOsoCalcGradients(VulkanCollection* collection, uint32_t id_grad_buf) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 4;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizesMax[0] * sizeof(float);
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

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };
				uint32_t specialization[2] = { SIZES[0] * SIZES[1], 32 };

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
				specializationInfo.pData = &specialization;



				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/oso_calc_gradients.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void recordOsoCalcGradientsAppend(VulkanCollection* collection, VkCommandBuffer* commandBuffer) {
			vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, collection[0].descriptorSets, 0, NULL);
			vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdDispatch(commandBuffer[0], (uint32_t)ceil(SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);

		}

		//LBFGS
		void initLBFGS() {
			vulkanLBFGS.n_lbfgs_memory = launchConfiguration.n_lbfgs_memory; // how many previous iterations are stored in the memory
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
			if (launchConfiguration.LBFGS_linesearch) {
				vulkanLBFGS.bufferSizes = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * (8));
				vulkanLBFGS.buffer = (VkBuffer*)malloc(sizeof(VkBuffer) * (8));
				vulkanLBFGS.deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) * (8));
			}
			else {
				vulkanLBFGS.bufferSizes = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * (7));
				vulkanLBFGS.buffer = (VkBuffer*)malloc(sizeof(VkBuffer) * (7));
				vulkanLBFGS.deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) * (7));
			}
			vulkanLBFGS.bufferSizes[0] = 3 * vulkanLBFGS.n_lbfgs_memory * n * sizeof(scalar);
			vulkanLBFGS.bufferSizes[1] = 3 * vulkanLBFGS.n_lbfgs_memory * n * sizeof(scalar);
			//vulkanLBFGS.bufferSizes[2] = vulkanLBFGS.n_lbfgs_memory * sizeof(float);
			//vulkanLBFGS.bufferSizes[3] = vulkanLBFGS.n_lbfgs_memory * sizeof(float);
			for (int i = 2; i < 5; i++)
			{
				vulkanLBFGS.bufferSizes[i] = 3 * n * sizeof(float);
			}
			vulkanLBFGS.bufferSizes[5] = 3 * n / 1024 * 2 * vulkanLBFGS.n_lbfgs_memory * sizeof(scalar);
			vulkanLBFGS.bufferSizes[6] = (3 * n / 1024 + 1) * sizeof(uint32_t);

			for (int i = 0; i < 7; i++)
			{
				allocateBuffer(&vulkanLBFGS.buffer[i], &vulkanLBFGS.deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanLBFGS.bufferSizes[i]);
			}
			if (launchConfiguration.LBFGS_linesearch) {
				vulkanLBFGS.bufferSizes[7] = 3 * n * sizeof(float);
				allocateBuffer(&vulkanLBFGS.buffer[7], &vulkanLBFGS.deviceMemory[7], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanLBFGS.bufferSizes[7]);

			}

		}
		void deleteLBFGS() {

			for (int i = 0; i < 7; i++)
			{
				vkFreeMemory(device, vulkanLBFGS.deviceMemory[i], NULL);
				vkDestroyBuffer(device, vulkanLBFGS.buffer[i], NULL);
			}
			if (launchConfiguration.LBFGS_linesearch) {
				vkFreeMemory(device, vulkanLBFGS.deviceMemory[7], NULL);
				vkDestroyBuffer(device, vulkanLBFGS.buffer[7], NULL);
			}
		}
		void createApplyLBFGS1(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 9;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
					collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[9];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 9; ++i) {
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
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[2];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[2];
					}
					if (i == 3) {

						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[3];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[3];
						
					}
					if (i == 5) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[5];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[5];
					}
					if (i == 6) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					if (i == 7) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = 2 * vulkanReduce.sizes[0] * sizeof(float);
					}

					if (i == 8) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 32 * sizeof(float);
						descriptorBufferInfo.range = 2 * sizeof(float);
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 6 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/ApplyLBFGS1.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}
			
		}
		void createApplyLBFGS2(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 9;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[9];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 9; ++i) {
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
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[2];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[2];
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
						
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[3];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[3];
					}
					if (i == 5) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[4];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[4];
					}
					if (i == 6) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[5];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[5];
					}
					if (i == 7) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					if (i == 8) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[vulkanReduce.bufferNum - 1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizes[vulkanReduce.bufferNum - 1] * sizeof(float);
					}


					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 6 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/ApplyLBFGS2.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyLBFGS3(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 4;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[4];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 4; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[4];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[4];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizes[0] * sizeof(float);
					}


					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 6 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/ApplyLBFGS3.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;
				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyLBFGS4(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 6;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[6];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 6; ++i) {
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
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[4];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[4];
					}
					if (i ==3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[5];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[5];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					if (i == 5) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[vulkanReduce.bufferNum - 1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = 2*vulkanReduce.sizes[vulkanReduce.bufferNum - 1] * sizeof(float);
					}


					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 6 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/ApplyLBFGS4.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyLBFGS5(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 4;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[4];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 4; ++i) {
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
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizes[0] * sizeof(float);
					}


					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 6 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/ApplyLBFGS5.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;
				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyLBFGS6(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 7;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[7];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 7; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[2];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[2];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
						
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[3];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[3];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[5];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[5];
					}
					if (i == 5) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					if (i == 6) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[vulkanReduce.bufferNum - 1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizes[vulkanReduce.bufferNum - 1] * sizeof(float);
					}


					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 6 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/ApplyLBFGS6.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyLBFGS7(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[2];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 2; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[2];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[2];
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizes[0] * sizeof(float);
					}


					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 6 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/ApplyLBFGS7.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;
				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyLBFGS8(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 6;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[6];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 6; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[2];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[2];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[6];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[6];
					}
					
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[vulkanReduce.bufferNum - 1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizes[vulkanReduce.bufferNum - 1] * sizeof(float);
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 32 * sizeof(float);
						descriptorBufferInfo.range = 2*sizeof(float);
					}
					if (i == 5) {
						if (launchConfiguration.LBFGS_linesearch) {
							descriptorBufferInfo.buffer = vulkanLBFGS.buffer[7];
							descriptorBufferInfo.offset = 0;
							descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[7];
						}
						else {
							descriptorBufferInfo.buffer = bufferSpins;
							descriptorBufferInfo.offset = 0;
							descriptorBufferInfo.range = bufferSizeSpins;
						}
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 6 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_rotate == true)
					code = readShader(filelength, "shaders/ApplyLBFGS8_double.spv");
				else
					code = readShader(filelength, "shaders/ApplyLBFGS8_float.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyLBFGS9(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 6;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[6];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 6; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[7];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[7];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[2];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[2];
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[3];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[3];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 5) {
						descriptorBufferInfo.buffer = bufferEnergy;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeEnergy;
					}
					
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{

				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 6 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_rotate == true)
					code = readShader(filelength, "shaders/ApplyLBFGS9_double.spv");
				else
					code = readShader(filelength, "shaders/ApplyLBFGS9_float.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void appendLBFGSstep(VulkanCollection* collection, VkCommandBuffer* commandBuffer) {
			int nos = SIZES[0] * SIZES[1] * SIZES[2];
			vkCmdPushConstants(commandBuffer[0], collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 6 * sizeof(uint32_t), &vulkanLBFGS.applyLBFGSConsts);
			vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(commandBuffer[0], (uint32_t)ceil(3 * SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);

		}
		void appendLBFGSstep2(VulkanCollection* collection, VkCommandBuffer* commandBuffer) {
			vkCmdPushConstants(commandBuffer[0], collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 6 * sizeof(uint32_t), &vulkanLBFGS.applyLBFGSConsts);
			vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(commandBuffer[0], (uint32_t)ceil( SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);

		}
		void recordFullBufferLBFGS(VkCommandBuffer* commandBuffer) {
			int nos = SIZES[0] * SIZES[1] * SIZES[2];
			//int groupedIterations = 1;// std::max(1, 8 * 1024 * 1024 / nos);
			VkMemoryBarrier memory_barrier= {
					VK_STRUCTURE_TYPE_MEMORY_BARRIER,
					nullptr,
					VK_ACCESS_SHADER_WRITE_BIT,
					VK_ACCESS_SHADER_READ_BIT,//VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
			};/*
			for (uint32_t i=0; i<8;i++)
			{
				buffer_barrier[i] = {
					VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
					nullptr,
					VK_ACCESS_MEMORY_WRITE_BIT,
					VK_ACCESS_MEMORY_READ_BIT,//VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
					queueFamilyIndex,
					queueFamilyIndex,
					vulkanLBFGS.buffer[i],
					0,
					vulkanLBFGS.bufferSizes[i]
				};
			}
			buffer_barrier[8]= {
					VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
					nullptr,
					VK_ACCESS_MEMORY_WRITE_BIT,
					VK_ACCESS_MEMORY_READ_BIT,//VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
					queueFamilyIndex,
					queueFamilyIndex,
					bufferSpins,
					0,
					bufferSizeSpins
			};
			buffer_barrier[9] = {
					VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
					nullptr,
					VK_ACCESS_MEMORY_WRITE_BIT,
					VK_ACCESS_MEMORY_READ_BIT,//VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
					queueFamilyIndex,
					queueFamilyIndex,
					vulkanReduce.buffer[vulkanReduce.bufferNum-1],
					0,
					2 * vulkanReduce.sizes[vulkanReduce.bufferNum - 1] * sizeof(float)
			};*/
			for (int i = 0; i < launchConfiguration.groupedIterations; i++) {
				if (launchConfiguration.DDI == true) {
					app_convolution.VkFFTAppend(commandBuffer[0]);
				}
				recordComputeGradients_noDDIAppend(&collectionGradients_noDDI_save, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				if (nos > 1024) {
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceEnergyFinish, commandBuffer, &vulkanReduce, &memory_barrier);
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceMaxFinish, commandBuffer, &vulkanReduce, &memory_barrier);

				}
				//recordOsoCalcGradientsAppend(&vulkanLBFGS.collectionOsoCalcGradients, commandBuffer);
				
				appendLBFGSstep(&vulkanLBFGS.collectionApply1, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				if (nos > 1024) {
					recordReduceDotFinishAppend(&vulkanLBFGS.collectionReduceDotFinish2, commandBuffer, &vulkanReduce, &memory_barrier);
				}
				appendLBFGSstep(&vulkanLBFGS.collectionApply2, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL); 
				for (uint32_t k = vulkanLBFGS.applyLBFGSConsts.num_mem -1; k > -1; k--) {
					vulkanLBFGS.applyLBFGSConsts.k = k;
					appendLBFGSstep(&vulkanLBFGS.collectionApply3, commandBuffer);
					vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
					if (nos > 1024) {
						recordReduceDotFinishAppend(&vulkanLBFGS.collectionReduceDotFinish, commandBuffer, &vulkanReduce, &memory_barrier);
					}
					appendLBFGSstep(&vulkanLBFGS.collectionApply4, commandBuffer);
					vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				}
				for (uint32_t k = 0; k < vulkanLBFGS.applyLBFGSConsts.num_mem; k++) {
					vulkanLBFGS.applyLBFGSConsts.k = k;
					appendLBFGSstep(&vulkanLBFGS.collectionApply5, commandBuffer);
					vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
					if (nos > 1024) {
						recordReduceDotFinishAppend(&vulkanLBFGS.collectionReduceDotFinish, commandBuffer, &vulkanReduce, &memory_barrier);
					}
					appendLBFGSstep(&vulkanLBFGS.collectionApply6, commandBuffer);
					vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				}
				
				appendLBFGSstep(&vulkanLBFGS.collectionApply7, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				if (nos > 1024) {
					recordReduceDotFinishAppend(&vulkanLBFGS.collectionReduceDotFinish3, commandBuffer, &vulkanReduce, &memory_barrier);
				}
				appendLBFGSstep2(&vulkanLBFGS.collectionApply8, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				/*if (nos > 1024) {
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceMaxFinish, commandBuffer, &vulkanReduce, &memory_barrier);
				}*/
				if (launchConfiguration.LBFGS_linesearch) {
					if (launchConfiguration.DDI == true) {
						app_convolution.VkFFTAppend(commandBuffer[0]);
					}
					recordComputeGradients_noDDIAppend(&collectionGradients_noDDI_save, commandBuffer);
					vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
					appendLBFGSstep2(&vulkanLBFGS.collectionApply9, commandBuffer);
					vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				}
			}
		}
		void createOsoRotate(VulkanCollection* collection, uint32_t id_buf_searchdir) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 2;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
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
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };
				uint32_t pad = SIZES[0] * SIZES[1] * SIZES[2];
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
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/oso_rotate.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
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
		
		void createCommandBufferFullLBFGS() {
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBufferFullLBFGS);
			VkCommandBufferBeginInfo commandBufferBeginInfo = {};
			commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			commandBufferBeginInfo.flags = NULL;
			VK_CHECK_RESULT(vkBeginCommandBuffer(commandBufferFullLBFGS, &commandBufferBeginInfo));
			recordFullBufferLBFGS(&commandBufferFullLBFGS);
			VK_CHECK_RESULT(vkEndCommandBuffer(commandBufferFullLBFGS));
		}
		void createCommandBufferEnergy() {
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collectionReduceDotEnergy.commandBuffer);
			VkCommandBufferBeginInfo commandBufferBeginInfo = {};
			commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			commandBufferBeginInfo.flags = NULL;
			VkMemoryBarrier memory_barrier = {
					VK_STRUCTURE_TYPE_MEMORY_BARRIER,
					nullptr,
					VK_ACCESS_SHADER_WRITE_BIT,
					VK_ACCESS_SHADER_READ_BIT,//VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
			};
			uint32_t nos = SIZES[0] * SIZES[1] * SIZES[2];
			VK_CHECK_RESULT(vkBeginCommandBuffer(collectionReduceDotEnergy.commandBuffer, &commandBufferBeginInfo));
			vkCmdPushConstants(collectionReduceDotEnergy.commandBuffer, collectionReduceDotEnergy.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(uint32_t), vulkanLBFGS.ReduceDotConsts);
			vkCmdBindPipeline(collectionReduceDotEnergy.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collectionReduceDotEnergy.pipelines[0]);
			vkCmdBindDescriptorSets(collectionReduceDotEnergy.commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collectionReduceDotEnergy.pipelineLayout, 0, 1, &collectionReduceDotEnergy.descriptorSets[0], 0, NULL);
			vkCmdDispatch(collectionReduceDotEnergy.commandBuffer, vulkanReduce.sizesMax[0], 1, 1);
			vkCmdPipelineBarrier(collectionReduceDotEnergy.commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
			if (nos > 1024) {
				recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceEnergyFinish, &collectionReduceDotEnergy.commandBuffer, &vulkanReduce, &memory_barrier);
			}
			VK_CHECK_RESULT(vkEndCommandBuffer(collectionReduceDotEnergy.commandBuffer));
		}
		void runLBFGS() {
			//int nos = SIZES[0] * SIZES[1] * SIZES[2];
			//auto time0 = std::chrono::steady_clock::now();
			/*if (iterations == 0) {

			}
			else {
				runCommandBuffer(&commandBufferFullVP);
			}*/
			vkQueueSubmit(queue, 1, &vulkanLBFGS.submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
			vkResetFences(device, 1, &fence);
			//auto time1 = std::chrono::steady_clock::now();
			//scalar projection_forcenorm[2];
			//runCommandBuffer(&vulkanLBFGS.collectionReduceDot0[0].commandBuffer);
			//if (nos > 1024)	runCommandBuffer(&vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
			//if (nos > 1024)	runCommandBuffer(&vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
			//auto time2 = std::chrono::steady_clock::now();
			//memcpy(projection_forcenorm, mapReduce, 2*sizeof(scalar));
			//auto time3 = std::chrono::steady_clock::now();
			//scalar force_norm = 1;
			//if (nos > 1024)	runCommandBuffer(&vulkanLBFGS.collectionReduceDotFinish2.commandBuffer);
			//if (nos > 1024)	runCommandBuffer(&vulkanLBFGS.collectionReduceDotFinish.commandBuffer);
			//auto time4 = std::chrono::steady_clock::now();
			//memcpy(&force_norm, mapReduce2, sizeof(scalar));
			//auto time5 = std::chrono::steady_clock::now();
			/*scalar ratio = projection_forcenorm[0] / projection_forcenorm[1];
			if (projection_forcenorm[0] <= 0) {
				vulkanLBFGS.applyVP2Consts.grad_mult = -dt * 0.5 * m_temp_inv;
			}
			else {
				vulkanLBFGS.applyVP2Consts.grad_mult = -dt * (ratio + 0.5 * m_temp_inv);
			}
			*/
			//recordApplyVP2(&vulkanLBFGS.collectionApply2);
			//runCommandBuffer(&vulkanLBFGS.collectionApply2.commandBuffer);
			//auto time6 = std::chrono::steady_clock::now();
			/*printf("Apply1: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time1 - time0).count() * 0.001);
			printf("collectionReduceDot0: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time2 - time1).count() * 0.001);
			printf("projection: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time3 - time2).count() * 0.001);
			printf("collectionReduceDot1: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time4 - time3).count() * 0.001);
			printf("projection: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time5 - time4).count() * 0.001);
			printf("Apply2: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time6 - time5).count() * 0.001);
			printf("all: %.3f ms\n", std::chrono::duration_cast<std::chrono::microseconds>(time6 - time0).count() * 0.001);*/
			//bufferTransfer(vulkanLBFGS.context, vulkanLBFGS.buffer[2], vulkanLBFGS.buffer[1], vulkanLBFGS.bufferSizes[1]);
			//runCommandBuffer(&commandBufferTransferSolver);


		}
		void createLBFGS(VulkanLBFGS* vulkanLBFGS) {
			uint32_t nos = SIZES[0] * SIZES[1] * SIZES[2];
			vulkanLBFGS->applyLBFGSConsts.num_mem = vulkanLBFGS->n_lbfgs_memory;
			vulkanLBFGS->applyLBFGSConsts.nos = nos;
			vulkanLBFGS->applyLBFGSConsts.pad = SIZES[0] * SIZES[1] * SIZES[2];
			vulkanLBFGS->applyLBFGSConsts.eps = 1e-30;
			vulkanLBFGS->applyLBFGSConsts.k = 0;
			vulkanLBFGS->applyLBFGSConsts.max_move = launchConfiguration.max_move;

			scalar maxmove_transfer[2] = {0,  launchConfiguration.max_move };
			
			void* map;
			vkMapMemory(device, vulkanReduce.deviceMemoryLastMax, 32 * sizeof(scalar), 2*sizeof(scalar), 0, &map);
			memcpy(map, maxmove_transfer, 2*sizeof(scalar));
			vkUnmapMemory(device, vulkanReduce.deviceMemoryLastMax);

			createReduceMaxFinish(&vulkanLBFGS->collectionReduceMaxFinish, &vulkanReduce, 1);
			createReduceDotFinishLBFGS(&vulkanLBFGS->collectionReduceDotFinish, &vulkanReduce, 1);
			createReduceDotFinishLBFGS(&vulkanLBFGS->collectionReduceDotFinish2, &vulkanReduce, 2);
			createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish3, &vulkanReduce, 1);
			createApplyLBFGS1(&vulkanLBFGS->collectionApply1);
			createApplyLBFGS2(&vulkanLBFGS->collectionApply2);
			/*vulkanLBFGS->collectionApply3 = (VulkanCollection*)malloc(vulkanLBFGS->n_lbfgs_memory * sizeof(VulkanCollection));
			vulkanLBFGS->collectionApply4 = (VulkanCollection*)malloc(vulkanLBFGS->n_lbfgs_memory * sizeof(VulkanCollection));
			vulkanLBFGS->collectionApply5 = (VulkanCollection*)malloc(vulkanLBFGS->n_lbfgs_memory * sizeof(VulkanCollection));
			vulkanLBFGS->collectionApply6 = (VulkanCollection*)malloc(vulkanLBFGS->n_lbfgs_memory * sizeof(VulkanCollection));*/

			createApplyLBFGS3(&vulkanLBFGS->collectionApply3);
			createApplyLBFGS4(&vulkanLBFGS->collectionApply4);
			createApplyLBFGS5(&vulkanLBFGS->collectionApply5);
			createApplyLBFGS6(&vulkanLBFGS->collectionApply6);
			createApplyLBFGS7(&vulkanLBFGS->collectionApply7);
			createApplyLBFGS8(&vulkanLBFGS->collectionApply8); 
			if (launchConfiguration.LBFGS_linesearch) 
				createApplyLBFGS9(&vulkanLBFGS->collectionApply9);
			//createOsoCalcGradients(&vulkanLBFGS->collectionOsoCalcGradients, 3);
			vulkanLBFGS->ReduceDotConsts[0] = nos;
			vulkanLBFGS->ReduceDotConsts[1] = 0;
			vulkanLBFGS->ReduceDotConsts[2] = 0;
			vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
			//createReduceEnergy(&collectionReduceDotEnergy, &bufferGradient, bufferSizeGradient, &bufferSpins, bufferSizeSpins);
			createReduceEnergyFinish(&vulkanLBFGS->collectionReduceEnergyFinish, &vulkanReduce, 8);


			createCommandBufferFullLBFGS();

			
			vulkanLBFGS->submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			vulkanLBFGS->submitInfo.commandBufferCount = 1; // submit a single command buffer
			vulkanLBFGS->submitInfo.pCommandBuffers = &commandBufferFullLBFGS; // the command buffer to submit.
			//createCommandBufferEnergy();
		};
		void deleteCollectionLBFGS(VulkanLBFGS* vulkanLBFGS) {
			
			deleteCollection(&vulkanLBFGS->collectionReduceMaxFinish);
			deleteCollection(&vulkanLBFGS->collectionReduceDotFinish);
			deleteCollection(&vulkanLBFGS->collectionReduceDotFinish2);
			deleteCollection(&vulkanLBFGS->collectionReduceDotFinish3);
			deleteCollection(&vulkanLBFGS->collectionApply1);
			deleteCollection(&vulkanLBFGS->collectionApply2);
			/*vulkanLBFGS->collectionApply3 = (VulkanCollection*)malloc(vulkanLBFGS->n_lbfgs_memory * sizeof(VulkanCollection));
			vulkanLBFGS->collectionApply4 = (VulkanCollection*)malloc(vulkanLBFGS->n_lbfgs_memory * sizeof(VulkanCollection));
			vulkanLBFGS->collectionApply5 = (VulkanCollection*)malloc(vulkanLBFGS->n_lbfgs_memory * sizeof(VulkanCollection));
			vulkanLBFGS->collectionApply6 = (VulkanCollection*)malloc(vulkanLBFGS->n_lbfgs_memory * sizeof(VulkanCollection));*/

			deleteCollection(&vulkanLBFGS->collectionApply3);
			deleteCollection(&vulkanLBFGS->collectionApply4);
			deleteCollection(&vulkanLBFGS->collectionApply5);
			deleteCollection(&vulkanLBFGS->collectionApply6);
			deleteCollection(&vulkanLBFGS->collectionApply7);
			deleteCollection(&vulkanLBFGS->collectionApply8);
			if (launchConfiguration.LBFGS_linesearch) 
				deleteCollection(&vulkanLBFGS->collectionApply9);
			//deleteCollection(&vulkanLBFGS->collectionOsoCalcGradients);

			//createReduceEnergy(&collectionReduceDotEnergy, &bufferGradient, bufferSizeGradient, &bufferSpins, bufferSizeSpins);
			deleteCollection(&vulkanLBFGS->collectionReduceEnergyFinish);

			vkFreeCommandBuffers(device, commandPool, 1, &commandBufferFullLBFGS);
			vulkanLBFGS = NULL;
			//createCommandBufferEnergy();
		};
		

		//VP
		void initVP() {
			int n = SIZES[0] * SIZES[1] * SIZES[2];

			/*vectorfield velocity = vectorfield(n, { 0,0,0 });
			vectorfield searchdir = vectorfield(n, { 0,0,0 });
			vectorfield grad = vectorfield(n, { 0,0,0 });
			vectorfield grad_pr = vectorfield(n, { 0,0,0 });*/

			//vulkanReduce.bufferNum++;
			vulkanLBFGS.bufferSizes = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) );

			for (int i = 0; i < 1; i++)
			{
				vulkanLBFGS.bufferSizes[i] = 3 * n * sizeof(float);
			}

			vulkanLBFGS.buffer = (VkBuffer*)malloc(sizeof(VkBuffer) );
			vulkanLBFGS.deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) );

			for (int i = 0; i < 1; i++)
			{
				allocateBuffer(&vulkanLBFGS.buffer[i], &vulkanLBFGS.deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanLBFGS.bufferSizes[i]);
			}

		}
		void deleteVP() {

			for (int i = 0; i < 1; i++)
			{
				vkFreeMemory(device, vulkanLBFGS.deviceMemory[i], NULL);
				vkDestroyBuffer(device, vulkanLBFGS.buffer[i], NULL);
			}
		}
		void createVP(VulkanLBFGS* vulkanLBFGS) {
			uint32_t nos = SIZES[0] * SIZES[1] * SIZES[2];
			createReduceVPFinish(&vulkanLBFGS->collectionReduceDotFinish, &vulkanReduce, 2);
			//createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish, &vulkanReduce, 1);
			//createOsoCalcGradients(&vulkanLBFGS->collectionOsoCalcGradients, 1);


			//vulkanLBFGS->applyVP2Consts.grad_mult = 0;
			vulkanLBFGS->applyVP2Consts.dt = launchConfiguration.gamma / 0.176085964411;
			vulkanLBFGS->applyVP2Consts.m_temp_inv = 0.5 * vulkanLBFGS->applyVP2Consts.dt / 0.01;
			vulkanLBFGS->applyVP2Consts.pad = SIZES[0] * SIZES[1] * SIZES[2];
			createApplyVP2(&vulkanLBFGS->collectionApply2);

			vulkanLBFGS->applyVP1Consts.m_temp_inv = vulkanLBFGS->applyVP2Consts.m_temp_inv;
			vulkanLBFGS->applyVP1Consts.n = nos;
			vulkanLBFGS->applyVP1Consts.pad = SIZES[0] * SIZES[1] * SIZES[2];

			createApplyVP1(&vulkanLBFGS->collectionApply1);
			
			scalar* data_grad_pr = (scalar*)malloc(3 * sizeof(scalar) * SIZES[0] * SIZES[1] * SIZES[2]);
			for (uint32_t i = 0; i < SIZES[0] * SIZES[1] * SIZES[2]; ++i) {
				data_grad_pr[i] = 0.0;
			}
			transferDataFromCPU(data_grad_pr, vulkanLBFGS->bufferSizes[0], &vulkanLBFGS->buffer[0]);
						
			scalar gamma_transfer[2];
			gamma_transfer[0] = 0;
			gamma_transfer[1] = 1;
			void* map;
			vkMapMemory(device, vulkanReduce.deviceMemoryLastMax, 32 * sizeof(scalar), 2 * sizeof(scalar), 0, &map);
			memcpy(map, gamma_transfer, 2 * sizeof(scalar));
			vkUnmapMemory(device, vulkanReduce.deviceMemoryLastMax);

			vulkanLBFGS->ReduceDotConsts[0] = nos;
			vulkanLBFGS->ReduceDotConsts[1] = 0;
			vulkanLBFGS->ReduceDotConsts[2] = 0;
			vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
			createReduceEnergyFinish(&vulkanLBFGS->collectionReduceEnergyFinish, &vulkanReduce, 8);
			createReduceMaxFinish(&vulkanLBFGS->collectionReduceMaxFinish, &vulkanReduce, 1);
			createCommandBufferFullVP();
			vulkanLBFGS->submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			vulkanLBFGS->submitInfo.commandBufferCount = 1; 
			vulkanLBFGS->submitInfo.pCommandBuffers = &commandBufferFullVP; 


		};
		void deleteCollectionVP(VulkanLBFGS* vulkanLBFGS) {
			deleteCollection(&vulkanLBFGS->collectionReduceDotFinish);
			deleteCollection(&vulkanLBFGS->collectionReduceMaxFinish);
			deleteCollection(&vulkanLBFGS->collectionApply2);
			deleteCollection(&vulkanLBFGS->collectionApply1);
			deleteCollection(&vulkanLBFGS->collectionReduceEnergyFinish);
			vkFreeCommandBuffers(device, commandPool, 1, &commandBufferFullVP);
			vulkanLBFGS = NULL;
		};
		void vp_get_searchdir(scalar m_temp_inv, scalar dt, uint32_t iterations) {

			vkQueueSubmit(queue, 1, &vulkanLBFGS.submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
			vkResetFences(device, 1, &fence);

		}
		void createCommandBufferFullVP() {
			
			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBufferFullVP);
			VkCommandBufferBeginInfo commandBufferBeginInfo = {};
			commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			commandBufferBeginInfo.flags = NULL;
			VK_CHECK_RESULT(vkBeginCommandBuffer(commandBufferFullVP, &commandBufferBeginInfo));
			recordFullBufferVP(&commandBufferFullVP);
			VK_CHECK_RESULT(vkEndCommandBuffer(commandBufferFullVP));
		}
		void createApplyVP2(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 4;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 32 * sizeof(float);
						descriptorBufferInfo.range = 2 * sizeof(float);
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
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 3 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };
				uint32_t specialization = 32;

				std::array<VkSpecializationMapEntry, 1> specializationMapEntries;
				specializationMapEntries[0].constantID = 1;
				specializationMapEntries[0].size = sizeof(uint32_t);
				specializationMapEntries[0].offset = 0;

				VkSpecializationInfo specializationInfo{};
				specializationInfo.dataSize = sizeof(uint32_t);
				specializationInfo.mapEntryCount = static_cast<uint32_t>(specializationMapEntries.size());
				specializationInfo.pMapEntries = specializationMapEntries.data();
				specializationInfo.pData = &specialization;



				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_rotate == true)
					code = readShader(filelength, "shaders/ApplyVP2_double.spv");
				else
					code = readShader(filelength, "shaders/ApplyVP2_float.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;


				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}
			{
				VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
				commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
				commandBufferAllocateInfo.commandPool = commandPool;
				commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				commandBufferAllocateInfo.commandBufferCount = 1;
				VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &collection[0].commandBuffer)); // allocate command buffer.
				vulkanLBFGS.applyVP2Consts.pad = SIZES[0] * SIZES[1] * SIZES[2];
				recordApplyVP2(collection);
			}
		}
		void recordApplyVP2(VulkanCollection* collection) {

			VkCommandBufferBeginInfo beginInfo = {};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			VK_CHECK_RESULT(vkBeginCommandBuffer(collection[0].commandBuffer, &beginInfo)); // start recording commands.
			vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(scalar), &vulkanLBFGS.applyVP2Consts);
			vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(collection[0].commandBuffer, (uint32_t)ceil(SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);
			VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.
		}
		void recordApplyVP2Append(VulkanCollection* collection, VkCommandBuffer* commandBuffer) {

			vkCmdPushConstants(commandBuffer[0], collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(scalar), &vulkanLBFGS.applyVP2Consts);
			vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(commandBuffer[0], (uint32_t)ceil(SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);

		}
		void recordFullBufferVP(VkCommandBuffer* commandBuffer) {
			int nos = SIZES[0] * SIZES[1] * SIZES[2];
			VkMemoryBarrier memory_barrier = {
					VK_STRUCTURE_TYPE_MEMORY_BARRIER,
					nullptr,
					VK_ACCESS_SHADER_WRITE_BIT,
					VK_ACCESS_SHADER_READ_BIT,//VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
			};
			//int groupedIterations = 1;// std::max(1, 8 * 1024 * 1024 / nos);
			for (int i = 0; i < launchConfiguration.groupedIterations; i++) {
				if (launchConfiguration.DDI == true) {
					app_convolution.VkFFTAppend(commandBuffer[0]);
				}
				recordComputeGradients_noDDIAppend(&collectionGradients_noDDI_save, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				if (nos > 1024) {
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceEnergyFinish, commandBuffer, &vulkanReduce, &memory_barrier);
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceMaxFinish, commandBuffer, &vulkanReduce, &memory_barrier);

				}


				recordApplyVP1Append(&vulkanLBFGS.collectionApply1, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				if (nos > 1024) {
					recordReduceDotFinishAppend(&vulkanLBFGS.collectionReduceDotFinish, commandBuffer, &vulkanReduce, &memory_barrier);
				}
				recordApplyVP2Append(&vulkanLBFGS.collectionApply2, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				/*if (nos > 1024) {
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceMaxFinish, commandBuffer, &vulkanReduce, &memory_barrier);
				}*/
			}
		}
		void createApplyVP1(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 4;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType[] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
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
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < COUNT_OF(descriptorType); ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = 2 * vulkanReduce.sizes[0] * sizeof(float);
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 32 * sizeof(float);
						descriptorBufferInfo.range = 2 * sizeof(float);
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
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 3 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code = readShader(filelength, "shaders/ApplyVP1.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
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
				vkCmdPushConstants(collection[0].commandBuffer, collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t), &vulkanLBFGS.applyVP1Consts);
				vkCmdBindPipeline(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
				vkCmdBindDescriptorSets(collection[0].commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
				vkCmdDispatch(collection[0].commandBuffer, vulkanReduce.sizes[0], 1, 1);
				VK_CHECK_RESULT(vkEndCommandBuffer(collection[0].commandBuffer)); // end recording commands.

			}
		}
		void recordApplyVP1Append(VulkanCollection* collection, VkCommandBuffer* commandBuffer) {
			vkCmdPushConstants(commandBuffer[0], collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 3 * sizeof(uint32_t), &vulkanLBFGS.applyVP1Consts);
			vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(commandBuffer[0], vulkanReduce.sizes[0], 1, 1);

		}



		//Depondt
		void initDepondt() {
			int n = SIZES[0] * SIZES[1] * SIZES[2];


			vulkanLBFGS.bufferSizes = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * (2));

			for (int i = 0; i < 2; i++)
			{
				vulkanLBFGS.bufferSizes[i] = 3 * n * sizeof(float);
			}

			vulkanLBFGS.buffer = (VkBuffer*)malloc(sizeof(VkBuffer) * (2));
			vulkanLBFGS.deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) * (2));

			for (int i = 0; i < 2; i++)
			{
				allocateBuffer(&vulkanLBFGS.buffer[i], &vulkanLBFGS.deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanLBFGS.bufferSizes[i]);
			}

		}
		void createDepondt(VulkanLBFGS* vulkanLBFGS) {
			uint32_t nos = SIZES[0] * SIZES[1] * SIZES[2];
			createReduceMaxFinish(&vulkanLBFGS->collectionReduceMaxFinish, &vulkanReduce, 1);
			//createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish, &vulkanReduce, 2);
			if (launchConfiguration.adaptiveTimeStep)
				vulkanLBFGS->applyDepondtConsts.dt = 0;
			else
				vulkanLBFGS->applyDepondtConsts.dt = launchConfiguration.gamma;
			vulkanLBFGS->applyDepondtConsts.pad = SIZES[0] * SIZES[1] * SIZES[2];
			scalar gamma_transfer[2];
			gamma_transfer[0]= 1e-5/launchConfiguration.gamma;
			gamma_transfer[1] = 0;
			void* map;
			vkMapMemory(device, vulkanReduce.deviceMemoryLastMax, 16*sizeof(scalar), 2*sizeof(scalar), 0, &map);
			memcpy(map, gamma_transfer, 2*sizeof(scalar));
			vkUnmapMemory(device, vulkanReduce.deviceMemoryLastMax);

			createApplyDepondt1(&vulkanLBFGS->collectionApply1);


			createApplyDepondt2(&vulkanLBFGS->collectionApply2);


			vulkanLBFGS->ReduceDotConsts[0] = nos;
			vulkanLBFGS->ReduceDotConsts[1] = 0;
			vulkanLBFGS->ReduceDotConsts[2] = 0;
			vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
			//createReduceEnergy(&collectionReduceDotEnergy, &bufferGradient, bufferSizeGradient, &bufferSpins, bufferSizeSpins);
			createReduceEnergyFinish(&vulkanLBFGS->collectionReduceEnergyFinish, &vulkanReduce, 8);

			createCommandBufferFullDepondt();
			vulkanLBFGS->submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			vulkanLBFGS->submitInfo.commandBufferCount = 1; // submit a single command buffer
			vulkanLBFGS->submitInfo.pCommandBuffers = &commandBufferFullDepondt; // the command buffer to submit.
			//createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish2, &vulkanReduce2);
			//createOsoCalcGradients(&vulkanLBFGS->collectionOsoCalcGradients, 1);
			//createOsoRotate(&vulkanLBFGS->collectionOsoRotate, 0);

		};
		void deleteDepondt() {

			for (int i = 0; i < 2; i++)
			{
				vkFreeMemory(device, vulkanLBFGS.deviceMemory[i], NULL);
				vkDestroyBuffer(device, vulkanLBFGS.buffer[i], NULL);
			}
		}
		void createApplyDepondt1(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 5;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[5];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 5; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 16 * sizeof(float);
						descriptorBufferInfo.range = sizeof(float);
					}

					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 2 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_rotate == true)
					code = readShader(filelength, "shaders/ApplyDepondt1_double.spv");
				else
					code = readShader(filelength, "shaders/ApplyDepondt1_float.spv");

				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyDepondt2(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 6;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[6];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 6; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanReduce.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanReduce.sizesMax[0] * sizeof(float);
					}
					if (i == 5) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 16* sizeof(float);
						descriptorBufferInfo.range = 2* sizeof(float);
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 2 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_rotate == true)
					code = readShader(filelength, "shaders/ApplyDepondt2_double.spv");
				else
					code = readShader(filelength, "shaders/ApplyDepondt2_float.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createCommandBufferFullDepondt() {

			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBufferFullDepondt);
			VkCommandBufferBeginInfo commandBufferBeginInfo = {};
			commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			commandBufferBeginInfo.flags = NULL;
			VK_CHECK_RESULT(vkBeginCommandBuffer(commandBufferFullDepondt, &commandBufferBeginInfo));
			recordFullBufferDepondt(&commandBufferFullDepondt);
			VK_CHECK_RESULT(vkEndCommandBuffer(commandBufferFullDepondt));
		}
		void recordFullBufferDepondt(VkCommandBuffer* commandBuffer) {
			int nos = SIZES[0] * SIZES[1] * SIZES[2];
			VkMemoryBarrier memory_barrier = {
					VK_STRUCTURE_TYPE_MEMORY_BARRIER,
					nullptr,
					VK_ACCESS_SHADER_WRITE_BIT,
					VK_ACCESS_SHADER_READ_BIT,//VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
			};
			//int groupedIterations = 1;// std::max(1, 8 * 1024 * 1024 / nos);
			for (int i = 0; i < launchConfiguration.groupedIterations; i++) {
				if (launchConfiguration.DDI == true) {
					app_convolution.VkFFTAppend(commandBuffer[0]);
				}
				recordComputeGradients_noDDIAppend(&collectionGradients_noDDI_save, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				if (nos > 1024) {
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceEnergyFinish, commandBuffer, &vulkanReduce, &memory_barrier);
				}
				recordApplyDepondtAppend(&vulkanLBFGS.collectionApply1, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				if (launchConfiguration.DDI == true) {
					app_convolution.VkFFTAppend(commandBuffer[0]);
				}
				recordComputeGradients_noDDIAppend(&collectionGradients_noDDI_nosave, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				recordApplyDepondtAppend(&vulkanLBFGS.collectionApply2, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				if (nos > 1024) {
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceMaxFinish, commandBuffer, &vulkanReduce, &memory_barrier);
				}
			}
		}
		void recordApplyDepondtAppend(VulkanCollection* collection, VkCommandBuffer* commandBuffer) {

			vkCmdPushConstants(commandBuffer[0], collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(scalar), &vulkanLBFGS.applyDepondtConsts);
			vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(commandBuffer[0], (uint32_t)ceil(SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);

		}
		void runDepondt() {

			vkQueueSubmit(queue, 1, &vulkanLBFGS.submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
			vkResetFences(device, 1, &fence);

		}
		void deleteCollectionDepondt(VulkanLBFGS* vulkanLBFGS) {
			deleteCollection(&vulkanLBFGS->collectionReduceMaxFinish);

			//vulkanLBFGS->applyVP2Consts.grad_mult = 0;

			deleteCollection(&vulkanLBFGS->collectionApply2);

			deleteCollection(&vulkanLBFGS->collectionApply1);





			//deleteCollection(&vulkanLBFGS->collectionReduceDot0);
			//deleteCollection(&vulkanLBFGS->collectionReduceDot1);
			//createReduceEnergy(&collectionReduceDotEnergy, &bufferGradient, bufferSizeGradient, &bufferSpins, bufferSizeSpins);
			deleteCollection(&vulkanLBFGS->collectionReduceEnergyFinish);

			vkFreeCommandBuffers(device, commandPool, 1, &commandBufferFullDepondt);
			vulkanLBFGS = NULL;
			//createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish2, &vulkanReduce2);
			//createOsoCalcGradients(&vulkanLBFGS->collectionOsoCalcGradients, 1);

		};

		//RK4
		void initRK4() {
			int n = SIZES[0] * SIZES[1] * SIZES[2];

			/*vectorfield velocity = vectorfield(n, { 0,0,0 });
			vectorfield searchdir = vectorfield(n, { 0,0,0 });
			vectorfield grad = vectorfield(n, { 0,0,0 });
			vectorfield grad_pr = vectorfield(n, { 0,0,0 });*/

			//vulkanReduce.bufferNum++;
			vulkanLBFGS.bufferSizes = (VkDeviceSize*)malloc(sizeof(VkDeviceSize) * (4));

			for (int i = 0; i < 4; i++)
			{
				vulkanLBFGS.bufferSizes[i] = 3 * n * sizeof(float);
			}

			vulkanLBFGS.buffer = (VkBuffer*)malloc(sizeof(VkBuffer) * (4));
			vulkanLBFGS.deviceMemory = (VkDeviceMemory*)malloc(sizeof(VkDeviceMemory) * (4));

			for (int i = 0; i < 4; i++)
			{
				allocateBuffer(&vulkanLBFGS.buffer[i], &vulkanLBFGS.deviceMemory[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, vulkanLBFGS.bufferSizes[i]);
			}

		}
		void createRK4(VulkanLBFGS* vulkanLBFGS) {
			uint32_t nos = SIZES[0] * SIZES[1] * SIZES[2];
			createReduceMaxFinish(&vulkanLBFGS->collectionReduceMaxFinish, &vulkanReduce, 1);
			//createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish, &vulkanReduce, 2);
			if (launchConfiguration.adaptiveTimeStep)
				vulkanLBFGS->applyRK4Consts.dt = 0;
			else
				vulkanLBFGS->applyRK4Consts.dt = launchConfiguration.gamma;
			vulkanLBFGS->applyRK4Consts.pad = SIZES[0] * SIZES[1] * SIZES[2];
			scalar gamma_transfer[2];
			gamma_transfer[0] = 1e-5 / launchConfiguration.gamma;
			gamma_transfer[1] = 0;
			void* map;
			vkMapMemory(device, vulkanReduce.deviceMemoryLastMax, 16 * sizeof(scalar), 2 * sizeof(scalar), 0, &map);
			memcpy(map, gamma_transfer, 2 * sizeof(scalar));
			vkUnmapMemory(device, vulkanReduce.deviceMemoryLastMax);

			createApplyRK4_1(&vulkanLBFGS->collectionApply1);


			createApplyRK4_2(&vulkanLBFGS->collectionApply2);
			createApplyRK4_3(&vulkanLBFGS->collectionApply3);
			createApplyRK4_4(&vulkanLBFGS->collectionApply4);

			vulkanLBFGS->ReduceDotConsts[0] = nos;
			vulkanLBFGS->ReduceDotConsts[1] = 0;
			vulkanLBFGS->ReduceDotConsts[2] = 0;
			vulkanLBFGS->ReduceDotConsts[3] = SIZES[0] * SIZES[1];
			//createReduceEnergy(&collectionReduceDotEnergy, &bufferGradient, bufferSizeGradient, &bufferSpins, bufferSizeSpins);
			createReduceEnergyFinish(&vulkanLBFGS->collectionReduceEnergyFinish, &vulkanReduce, 8);

			createCommandBufferFullRK4();
			vulkanLBFGS->submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			vulkanLBFGS->submitInfo.commandBufferCount = 1; // submit a single command buffer
			vulkanLBFGS->submitInfo.pCommandBuffers = &commandBufferFullRK4; // the command buffer to submit.
			//createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish2, &vulkanReduce2);
			//createOsoCalcGradients(&vulkanLBFGS->collectionOsoCalcGradients, 1);
			//createOsoRotate(&vulkanLBFGS->collectionOsoRotate, 0);

		};
		void deleteRK4() {

			for (int i = 0; i < 4; i++)
			{
				vkFreeMemory(device, vulkanLBFGS.deviceMemory[i], NULL);
				vkDestroyBuffer(device, vulkanLBFGS.buffer[i], NULL);
			}
		}
		void createApplyRK4_1(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 5;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[5];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 5; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 16 * sizeof(float);
						descriptorBufferInfo.range = sizeof(float);
					}

					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 2 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_rotate == true)
					code = readShader(filelength, "shaders/ApplyRK4_1_double.spv");
				else
					code = readShader(filelength, "shaders/ApplyRK4_1_float.spv");

				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyRK4_2(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 5;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[5];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 5; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[2];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[2];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 16 * sizeof(float);
						descriptorBufferInfo.range = sizeof(float);
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 2 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_rotate == true)
					code = readShader(filelength, "shaders/ApplyRK4_2_double.spv");
				else
					code = readShader(filelength, "shaders/ApplyRK4_2_float.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;

				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyRK4_3(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 5;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[5];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 5; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[3];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[3];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 16 * sizeof(float);
						descriptorBufferInfo.range = sizeof(float);
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 2 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_rotate == true)
					code = readShader(filelength, "shaders/ApplyRK4_3_double.spv");
				else
					code = readShader(filelength, "shaders/ApplyRK4_3_float.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createApplyRK4_4(VulkanCollection* collection) {
			{
				VkDescriptorPoolSize descriptorPoolSize[1] = { };
				descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				descriptorPoolSize[0].descriptorCount = 7;
				//collection->descriptorNum = descriptorPoolSize[0].descriptorCount;

				VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
				descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
				descriptorPoolCreateInfo.poolSizeCount = COUNT_OF(descriptorPoolSize);
				descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;
				descriptorPoolCreateInfo.maxSets = 1;
				vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &collection[0].descriptorPool);
			}

			{
				const VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				collection[0].descriptorSetLayouts = (VkDescriptorSetLayout*)malloc(sizeof(VkDescriptorSetLayout));
				collection[0].descriptorSets = (VkDescriptorSet*)malloc(sizeof(VkDescriptorSet));
				collection->descriptorNum = 1;
				VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[7];
				for (uint32_t i = 0; i < COUNT_OF(descriptorSetLayoutBindings); ++i) {
					descriptorSetLayoutBindings[i].binding = i;
					descriptorSetLayoutBindings[i].descriptorType = descriptorType;
					descriptorSetLayoutBindings[i].descriptorCount = 1;
					descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				}
				VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
				descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
				descriptorSetLayoutCreateInfo.bindingCount = COUNT_OF(descriptorSetLayoutBindings);
				descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings;
				vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, collection[0].descriptorSetLayouts);
				VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = { };
				descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				descriptorSetAllocateInfo.descriptorPool = collection[0].descriptorPool;
				descriptorSetAllocateInfo.descriptorSetCount = 1;
				descriptorSetAllocateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, collection[0].descriptorSets);
				for (uint32_t i = 0; i < 7; ++i) {
					VkDescriptorBufferInfo descriptorBufferInfo = { };
					if (i == 0) {
						descriptorBufferInfo.buffer = bufferSpins;
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = bufferSizeSpins;
					}
					if (i == 1) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[0];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[0];
					}
					if (i == 2) {
						descriptorBufferInfo.buffer = bufferGradient;
						descriptorBufferInfo.offset = bufferSizeGradient;
						descriptorBufferInfo.range = bufferSizeGradient;
					}
					if (i == 3) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[1];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[1];
					}
					if (i == 4) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[2];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[2];
					}
					if (i == 5) {
						descriptorBufferInfo.buffer = vulkanLBFGS.buffer[3];
						descriptorBufferInfo.offset = 0;
						descriptorBufferInfo.range = vulkanLBFGS.bufferSizes[3];
					}
					if (i == 6) {
						descriptorBufferInfo.buffer = vulkanReduce.lastMax;
						descriptorBufferInfo.offset = 16 * sizeof(float);
						descriptorBufferInfo.range = sizeof(float);
					}
					VkWriteDescriptorSet writeDescriptorSet = { };
					writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					writeDescriptorSet.dstSet = collection[0].descriptorSets[0];
					writeDescriptorSet.dstBinding = i;
					writeDescriptorSet.dstArrayElement = 0;
					writeDescriptorSet.descriptorType = descriptorType;
					writeDescriptorSet.descriptorCount = 1;
					writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;
					vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
				}
			}

			{
				
				collection[0].pipelines = (VkPipeline*)malloc(sizeof(VkPipeline));
				VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = { };
				pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
				pipelineLayoutCreateInfo.setLayoutCount = 1;
				pipelineLayoutCreateInfo.pSetLayouts = collection[0].descriptorSetLayouts;
				VkPushConstantRange pushConstantRange = {};
				pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
				pushConstantRange.offset = 0;
				pushConstantRange.size = 2 * sizeof(scalar);
				// Push constant ranges are part of the pipeline layout
				pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
				pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
				vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &collection[0].pipelineLayout);
				VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo = { };
				VkComputePipelineCreateInfo computePipelineCreateInfo = { };

				struct SpecializationData {
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

				pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
				pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
				uint32_t filelength;
				uint32_t* code;
				if (launchConfiguration.double_precision_rotate == true)
					code = readShader(filelength, "shaders/ApplyRK4_4_double.spv");
				else
					code = readShader(filelength, "shaders/ApplyRK4_4_float.spv");
				VkShaderModuleCreateInfo createInfo = {};
				createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
				createInfo.pCode = code;
				createInfo.codeSize = filelength;
				vkCreateShaderModule(device, &createInfo, NULL, &pipelineShaderStageCreateInfo.module);
				delete[] code;
				pipelineShaderStageCreateInfo.pName = "main";
				pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;
				computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
				computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
				computePipelineCreateInfo.layout = collection[0].pipelineLayout;



				vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &computePipelineCreateInfo, NULL, collection[0].pipelines);
				vkDestroyShaderModule(device, pipelineShaderStageCreateInfo.module, NULL);
			}

		}
		void createCommandBufferFullRK4() {

			VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
			commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			commandBufferAllocateInfo.commandPool = commandPool;
			commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			commandBufferAllocateInfo.commandBufferCount = 1;
			vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBufferFullRK4);
			VkCommandBufferBeginInfo commandBufferBeginInfo = {};
			commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			commandBufferBeginInfo.flags = NULL;
			VK_CHECK_RESULT(vkBeginCommandBuffer(commandBufferFullRK4, &commandBufferBeginInfo));
			recordFullBufferRK4(&commandBufferFullRK4);
			VK_CHECK_RESULT(vkEndCommandBuffer(commandBufferFullRK4));
		}
		void recordFullBufferRK4(VkCommandBuffer* commandBuffer) {
			int nos = SIZES[0] * SIZES[1] * SIZES[2];
			VkMemoryBarrier memory_barrier = {
					VK_STRUCTURE_TYPE_MEMORY_BARRIER,
					nullptr,
					VK_ACCESS_SHADER_WRITE_BIT,
					VK_ACCESS_SHADER_READ_BIT,//VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
			};
			//int groupedIterations = 1;// std::max(1, 8 * 1024 * 1024 / nos);
			for (int i = 0; i < launchConfiguration.groupedIterations; i++) {
				if (launchConfiguration.DDI == true) {
					app_convolution.VkFFTAppend(commandBuffer[0]);
				}
				recordComputeGradients_noDDIAppend(&collectionGradients_noDDI_save, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);
				if (nos > 1024) {
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceEnergyFinish, commandBuffer, &vulkanReduce, &memory_barrier);
					recordReduceMaxFinishAppend(&vulkanLBFGS.collectionReduceMaxFinish, commandBuffer, &vulkanReduce, &memory_barrier);

				}
				recordApplyRK4Append(&vulkanLBFGS.collectionApply1, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				if (launchConfiguration.DDI == true) {
					app_convolution.VkFFTAppend(commandBuffer[0]);
				}
				recordComputeGradients_noDDIAppend(&collectionGradients_noDDI_nosave, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				recordApplyRK4Append(&vulkanLBFGS.collectionApply2, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				if (launchConfiguration.DDI == true) {
					app_convolution.VkFFTAppend(commandBuffer[0]);
				}
				recordComputeGradients_noDDIAppend(&collectionGradients_noDDI_nosave, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				recordApplyRK4Append(&vulkanLBFGS.collectionApply3, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				if (launchConfiguration.DDI == true) {
					app_convolution.VkFFTAppend(commandBuffer[0]);
				}
				recordComputeGradients_noDDIAppend(&collectionGradients_noDDI_nosave, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);

				recordApplyRK4Append(&vulkanLBFGS.collectionApply4, commandBuffer);
				vkCmdPipelineBarrier(commandBuffer[0], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memory_barrier, 0, NULL, 0, NULL);




			}
		}
		void recordApplyRK4Append(VulkanCollection* collection, VkCommandBuffer* commandBuffer) {

			vkCmdPushConstants(commandBuffer[0], collection[0].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(scalar), &vulkanLBFGS.applyRK4Consts);
			vkCmdBindPipeline(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelines[0]);
			vkCmdBindDescriptorSets(commandBuffer[0], VK_PIPELINE_BIND_POINT_COMPUTE, collection[0].pipelineLayout, 0, 1, &collection[0].descriptorSets[0], 0, NULL);
			vkCmdDispatch(commandBuffer[0], (uint32_t)ceil(SIZES[0] * SIZES[1] * SIZES[2] / 1024.0f), 1, 1);

		}
		void runRK4() {

			vkQueueSubmit(queue, 1, &vulkanLBFGS.submitInfo, fence);
			vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000);
			vkResetFences(device, 1, &fence);

		}
		void deleteCollectionRK4(VulkanLBFGS* vulkanLBFGS) {
			deleteCollection(&vulkanLBFGS->collectionReduceMaxFinish);


			//vulkanLBFGS->applyVP2Consts.grad_mult = 0;



			deleteCollection(&vulkanLBFGS->collectionApply1);
			deleteCollection(&vulkanLBFGS->collectionApply2);
			deleteCollection(&vulkanLBFGS->collectionApply3);
			deleteCollection(&vulkanLBFGS->collectionApply4);




			//deleteCollection(&vulkanLBFGS->collectionReduceDot0);
			//deleteCollection(&vulkanLBFGS->collectionReduceDot1);
			//createReduceEnergy(&collectionReduceDotEnergy, &bufferGradient, bufferSizeGradient, &bufferSpins, bufferSizeSpins);
			deleteCollection(&vulkanLBFGS->collectionReduceEnergyFinish);

			vkFreeCommandBuffers(device, commandPool, 1, &commandBufferFullRK4);
			vulkanLBFGS = NULL;
			//createReduceDotFinish(&vulkanLBFGS->collectionReduceDotFinish2, &vulkanReduce2);
			//createOsoCalcGradients(&vulkanLBFGS->collectionOsoCalcGradients, 1);

		};

		//Clean
		void deleteCollection(VulkanCollection* collection) {
			//vkFreeDescriptorSets(device, collection->descriptorPool, collection->descriptorNum, collection->descriptorSets);
			//if (collection->commandBuffer != NULL)
				//vkFreeCommandBuffers(device, commandPool, 1, &collection->commandBuffer);

			vkDestroyDescriptorPool(device, collection->descriptorPool, NULL);
			vkDestroyDescriptorSetLayout(device, collection->descriptorSetLayouts[0],NULL);
			vkDestroyPipelineLayout(device, collection->pipelineLayout, NULL);
			vkDestroyPipeline(device, collection->pipelines[0], NULL);
			collection = NULL;
			
		}
		void cleanup() {
			/*
			Clean up all Vulkan Resources.
			*/
			if (enableValidationLayers) {
				DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
			}

			if (launchConfiguration.solver_type == 0) deleteLBFGS();
			if (launchConfiguration.solver_type == 1) deleteVP();
			if (launchConfiguration.solver_type == 2) deleteDepondt();
			if (launchConfiguration.solver_type == 3) deleteRK4();

			if (launchConfiguration.DDI == 1) {
				app_kernel.deleteVulkanFFT();
				app_convolution.deleteVulkanFFT();
			}
			deleteReduceBuffers(&vulkanReduce);
			vkDestroyBuffer(device, bufferSpins, NULL);
			vkDestroyBuffer(device, bufferGradient, NULL);
			vkDestroyBuffer(device, bufferStagingSpins, NULL);
			vkDestroyBuffer(device, bufferStagingGradient, NULL);
			vkDestroyBuffer(device, bufferRegions_Book, NULL);
			vkDestroyBuffer(device, bufferRegions, NULL);
			vkDestroyBuffer(device, bufferSpinsInit, NULL);
			vkDestroyBuffer(device, bufferGradientOut, NULL);
			vkDestroyBuffer(device, uboDimensions, NULL);
			vkDestroyBuffer(device, bufferEnergy, NULL);

			if (launchConfiguration.DDI) {
				vkDestroyBuffer(device, kernel, NULL);
				vkFreeMemory(device, bufferMemoryKernel, NULL);
				vkDestroyBuffer(device, bufferFFT, NULL);
				vkFreeMemory(device, bufferMemoryFFT, NULL);
			}

			vkFreeMemory(device, bufferMemorySpins, NULL);
			vkFreeMemory(device, bufferMemoryGradient, NULL);
			vkFreeMemory(device, bufferMemoryStagingSpins, NULL);
			vkFreeMemory(device, bufferMemoryStagingGradient, NULL);
			vkFreeMemory(device, bufferMemoryRegions_Book, NULL);
			vkFreeMemory(device, bufferMemoryRegions, NULL);
			vkFreeMemory(device, bufferMemorySpinsInit, NULL);
			vkFreeMemory(device, bufferMemoryGradientOut, NULL);
			vkFreeMemory(device, uboMemoryDimensions, NULL);
			vkFreeMemory(device, bufferMemoryEnergy, NULL);
			freeLastSolver();
			deleteCollection(&collectionReadSpins);
			deleteCollection(&collectionWriteGradient);
			deleteCollection(&collectionWriteSpins);
			deleteCollection(&collectionGradients_noDDI_nosave);
			deleteCollection(&collectionGradients_noDDI_save);
			vkDestroyFence(device, fence, NULL);
			vkDestroyCommandPool(device, commandPool, NULL);
			vkDestroyDevice(device, NULL);
			vkDestroyInstance(instance, NULL);
		}
	};
}


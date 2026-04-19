<?php

use Bitrix\Main\Page\Asset;
use Bitrix\Main\Loader;
use Bitrix\Highloadblock as HL;
// use CCrmProductRow;
use Bitrix\Crm\DealTable;
// use CUserFieldEnum;
use Bitrix\Crm\ContactTable;
use Bitrix\Crm\FieldMultiTable;
use Gtd\Finist\AgentTable;
use Gtd\Finist\DeliveryTable;
use Gtd\Finist\Doc\DocumentTable;

Loader::includeModule('gtd.finist');
Loader::includeModule('highloadblock');
Loader::includeModule('crm');

class CrmDealDelivery extends CBitrixComponent
{

	//222222
	//333333333333333

    public $jsonResponse = [];
    private $validationMessage = [];
    private $postData = [];
    private $dealData = [];
    private $sendData2Server = [];


    private $sendPackegeMap = [
        'baseName' => [
            'FIELD' => 'UF_CRM_5B33779D7F3B7',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'autoName' => [
            'FIELD' => 'UF_CRM_5B97638CBCC63',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'Ifl' => [
            'FIELD' => 'UF_CRM_5B71D0BF80ABF',
            'TYPE' => 'ARRAY',
            'FROM' => 'DEAL'
        ],
        'KaskoUnderwriter' => [
            'FIELD' => 'UF_CRM_1530177224',
            'TYPE' => 'USER',
            'FROM' => 'DEAL'
        ],
        'OsagoUnderwriter' => [
            'FIELD' => 'UF_CRM_1530177249',
            'TYPE' => 'USER',
            'FROM' => 'DEAL'
        ],
        'SNBsoKasko' => [
            'FIELD' => 'UF_CRM_1536917510',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'SNKvitOsago' => [
            'FIELD' => 'UF_CRM_1536919892',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'SNKvitKasko' => [
            'FIELD' => 'UF_CRM_1536919770',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'SNKvitIS' => [
            'FIELD' => 'UF_CRM_1589791140',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'SNPolisKasko' => [
            'FIELD' => 'UF_CRM_1536917510',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'SNPolisOsago' => [
            'FIELD' => 'UF_CRM_1532343502',
            'TYPE' => "STRING",
            'FROM' => 'DEAL'
        ],
        'SNPolisIFL' => [
            'FIELD' => 'UF_CRM_1536917528',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'SNBsoIfl' => [
            'FIELD' => 'UF_CRM_1536917483',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'SNIul' => [
            'FIELD' => 'UF_CRM_MY_STRING22',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'SNDms' => [
            'FIELD' => 'UF_CRM_CUSTOM_DMS',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'SNKvitIfl' => [
            'FIELD' => 'UF_CRM_1536919812',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'Roditdeal' => [
            'FIELD' => 'UF_CRM_1621867240662',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'Ofskidka' => [
            'FIELD' => 'UF_CRM_1622474829932',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'Anderskidka' => [
            'FIELD' => 'UF_CRM_1724336577',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL'
        ],
        'Deal' => [
            'FIELD' => 'ID',
            'TYPE' => "STRING",
            'FROM' => 'DEAL'
        ],
        'Category' => [
            'FIELD' => 'CATEGORY_ID',
            'TYPE' => "STRING",
            'FROM' => 'DEAL'
        ],
        'City' => [
            'FIELD' => 'CITY',
            'FUNC' => 'getCity',
            'FROM' => 'POST'
        ],
        'Metro' => [
            'FIELD' => 'METRO',
            'FUNC' => 'getMetro',
            'FROM' => 'POST'
        ],
        'Address' => [
            'FIELD' => 'ADDRESS',
            'TYPE' => "STRING",
            'FROM' => 'POST'
        ],
        'Date' => [
            'FIELD' => 'DATE',
            'TYPE' => "STRING",
            'FROM' => 'POST'
        ],
        'TimeFrom' => [
            'FIELD' => 'TIME_FROM',
            'TYPE' => "STRING",
            'FROM' => 'POST'
        ],
        'TimeTill' => [
            'FIELD' => 'TIME_TO',
            'TYPE' => "STRING",
            'FROM' => 'POST'
        ],
        'Client' => [
            'FIELD' => 'CONTACT_ID',
            'FUNC' => 'getClientInfo',
            'FROM' => 'DEAL'
        ],
        'Responsible' => [
            'FIELD' => 'ASSIGNED_BY_ID',
            'FUNC' => 'getRespInfo',
            'FROM' => 'DEAL'
        ],
        'ResponsibleChief' => [
            'FIELD' => 'ASSIGNED_BY_ID',
            'FUNC' => 'getChiefForUser',
            'FROM' => 'DEAL'
        ],
        'UnderwriterIfl' => [
            'FIELD' => "UF_CRM_1530177276",
            'FUNC' => 'getRespInfo',
            'FROM' => "DEAL"
        ],
        'UnderwriterKasco' => [
            'FIELD' => "UF_CRM_1530177224",
            'FUNC' => 'getRespInfo',
            'FROM' => "DEAL"
        ],
        'UnderwriterOsago' => [
            'FIELD' => "UF_CRM_1530177249",
            'FUNC' => 'getRespInfo',
            'FROM' => "DEAL"
        ],
        'Product' => [
            'FIELD' => 'ID',
            'FUNC' => 'getProduct'
        ],
        'Type' => [
            'FIELD' => 'TYPE',
            'FROM' => 'POST',
            'TYPE' => 'STRING'
        ],
        'Inspection' => [
            'FIELD' => 'OSMOTR',
            'FROM' => 'POST',
            'TYPE' => 'STRING'
        ],
        'Comment' => [
            'FIELD' => 'COMMENT',
            'FROM' => 'POST',
            'TYPE' => "STRING",
            'FUNC' => 'getComment'
        ],
        'PaidDeparture' => [
            'FIELD' => 'PD',
            'FROM' => "POST",
            'TYPE' => "STRING"
        ],
        'PaidDeparturePrice' => [
            'FIELD' => 'PD_SUM',
            'FROM' => 'POST',
            'TYPE' => 'STRING'
        ],
        'SNBSODGO' => [
            'FIELD' => 'UF_CRM_1541062224',
            'FROM' => "DEAL",
            'TYPE' => "STRING"
        ],
        'SNBSOKvitDGO' => [
            'FIELD' => 'UF_CRM_1541066552',
            'FROM' => "DEAL",
            'TYPE' => "STRING"
        ],
        'SNDK' => [
            'FIELD' => 'UF_CRM_1541062250',
            'FROM' => "DEAL",
            'TYPE' => "STRING"
        ],
        'PlaceOfSale' => [
            'FIELD' => 'SALE_PLACE',
            'FROM' => 'POST',
            'TYPE' => 'STRING'
        ],
        'SNPolisNC' => [
            'FIELD' => "UF_CRM_1548935125",
            'TYPE' => 'STRING',
            'FROM' => "DEAL"
        ],
        'SNKvitNC' => [
            'FIELD' => "UF_CRM_1548935146",
            'TYPE' => 'STRING',
            'FROM' => "DEAL"
        ],
        'SNPolisVZP' => [
            'FIELD' => "UF_CRM_1548935376",
            'TYPE' => 'STRING',
            'FROM' => "DEAL"
        ],
        'SNBSOVZP' => [
            'FIELD' => "UF_CRM_1548935404",
            'TYPE' => 'STRING',
            'FROM' => "DEAL"
        ],
        'SNKvitVZP' => [
            'FIELD' => "UF_CRM_1548935433",
            'TYPE' => 'STRING',
            'FROM' => "DEAL"
        ],
        'SNPolisGAP' => [
            'FIELD' => "UF_CRM_1548935462",
            'TYPE' => 'STRING',
            'FROM' => "DEAL"
        ],
        'SNKvitGAP' => [
            'FIELD' => "UF_CRM_1548935481",
            'TYPE' => 'STRING',
            'FROM' => "DEAL"
        ],
        'SNPolisZK' => [
            'FIELD' => "UF_CRM_1549040778",
            'TYPE' => "STRING",
            "FROM" => "DEAL"
        ],
        "SNKvitZK" => [
            'FIELD' => 'UF_CRM_1549040796',
            'TYPE' => 'STRING',
            'FROM' => "DEAL"
        ],
        "SNPolisISIM" => [
            'FIELD' => 'UF_CRM_63A15C7438175',
            'TYPE' => 'STRING',
            'FROM' => "DEAL"
        ],
        'agent' => [
            'FIELD' => 'AGENT',
            'TYPE' => 'STRING',
            'FROM' => 'POST'
        ],
        'Bank' => [
            'FIELD' => 'UF_CRM_5B62FC4D4FAEC',
            'FROM' => "DEAL",
            'TYPE' => "ENUM",
        ],
        'SNPolisIS' => [
            'FIELD' => 'UF_CRM_1562757630',
            'FROM' => "DEAL",
            'TYPE' => "STRING"
        ],
        'Source' => [
            'FIELD' => 'SOURCE_ID',
            'FROM' => 'DEAL',
            'TYPE' => "STRING",
            'FUNC' => 'getSource'
        ],
        'DiscountFromHFPartner' => [
            'FIELD' => 'UF_CRM_1588249648',
            'FROM' => 'DEAL',
            'TYPE' => 'STRING'
        ],
        'ConsultingCost' => [
            'FIELD' => 'CONSULTING_COST',
            'TYPE' => 'STRING',
            'FROM' => 'POST',
        ],
        'Imushestvo' => [
            'FIELD' => 'UF_CRM_IM_BITRIX24',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL',
        ],
        'Titul' => [
            'FIELD' => 'UF_CRM_1647937084232',
            'TYPE' => 'STRING',
            'FROM' => 'DEAL',
        ],
    ];

	//22222222222222222222
	//3333333333333333333333333333333

    private $validation = [
        [
            'FIELD' => 'Source',
            'FUNC' => 'validateSource',
            'MESSAGE' => "Не заполнено поле в сделке AgentId"
        ]
    ];

    public function getAgentDealFieldName()
    {
        return COption::GetOptionString('gtd.finist', 'DEAL_AGENT_FIELD', 'F00000010000');
    }


    /**
     * @param $id
     * @return \Bitrix\Main\ORM\Data\DataManager
     * @throws \Bitrix\Main\ArgumentException
     * @throws \Bitrix\Main\ObjectPropertyException
     * @throws \Bitrix\Main\SystemException
     */
    public static function getHlClass($id)
    {
        $hlblock = HL\HighloadBlockTable::getById($id)->fetch();
        $entity = HL\HighloadBlockTable::compileEntity($hlblock);
        return $entity->getDataClass();
    }

    public function loadProduct()
    {
        $this->arResult['PRODUCT'] = CCrmProductRow::LoadRows('D', $this->arParams['ID'], false);
    }

    public function getTypeDelivery()
    {
        return [
            ['name' => 'Продажа', 'id' => 1],
            ['name' => 'Осмотр', 'id' => 2],
            ['name' => 'Перепись полиса', 'id' => 3],
            ['name' => 'Добор документов', 'id' => 4],
            ['name' => 'Поездка в страховую', 'id' => 5],
            //['name' => 'Доставка ДАБ', 'id' => 6],
            ['name' => 'Аддендум', 'id' => 8],
            ['name' => 'Осмотр оценка', 'id' => 9],
            ['name' => 'Доставка оценка', 'id' => 10],
        ];
    }

    public function getInspection()
    {
        return [
            "не требуется",
            "осмотр ТС",
            "осмотр дома",
            "осмотр квартиры",
            "осмотр имущества юр. лица",
            "осмотр ТС представителем СК"
        ];
    }

    public function getCityByMetroCode($code)
    {
        $cityClass = self::getHlClass(1);
        $metroClass = self::getHlClass(2);
        $resMetro = $metroClass::getList(['filter' => ['=UF_XML_ID' => $code], 'select' => ['UF_CITY']])->fetch();
        if ($resMetro) {
            return $resMetro['UF_CITY'];
        }
        return "";
    }

    public function getMetroArray()
    {
        $cityClass = self::getHlClass(1);
        $metroClass = self::getHlClass(2);

        $cityObj = $cityClass::getList(['filter' => ['=UF_ACTIVE' => true]]);
        $arrMetro = $metroClass::getList(['order' => array('UF_SORT' => 'ASC'), ])->fetchAll();

        while ($resCity = $cityObj->fetch()) {
            $arCity['city'][] = [
                'name' => $resCity['UF_NAME'],
                'id' => (int)$resCity['ID']
            ];

            foreach ($arrMetro as $arr) {
                if ($arr['UF_CITY'] == $resCity['ID']) {
                    $arCity['metro'][$resCity['ID']][] = $arr;
                }
            }
        }

        return $arCity;
    }


    public function ajaxPostData()
    {
        $this->dealData = $this->getDealData();
        $this->postData = $this->getPostBody();

        //отправка данных с формы "Выход на сделку"
        if ($this->postData['sendex'] == 'yessendex') {
            $sendData3Server = $this->postData;
            $connect1 = new \Gtd\Finist\UTP();
            $response1 = $connect1->SendRestRequest('getdoc2', '', true, true, $sendData3Server);
            if ($response1['guid']) {
                $this->jsonResponse['success'] = true;
            } else {
                $this->jsonResponse['success'] = false;
                $this->jsonResponse['message'] = $response1['errorMessage'];
                Sentry\captureMessage('не удалось отправить в доставку');
            }
        } else if ($this->postData['sto'] == true) { // Отправляем сделку в базу 1С СТО через SMTP
            $mail = new PHPMailer\PHPMailer\PHPMailer();
            $data2Mailbox = json_encode($this->postData, true);
            $filename = "/tmp/{$this->postData['BitrixId']}.json";

            try {

                file_put_contents($filename, $data2Mailbox);

                $mail->IsSMTP();
                $mail->SMTPAuth      = true;
                $mail->SMTPKeepAlive = true;
                //$mail->SMTPDebug = 1;
                $mail->SMTPSecure = 'tls';
                //   $mail->Host = 'mail-server.finist.ru';
                $mail->Host = 'smtp.gmail.com';
                $mail->Port = 587; // 587
                $mail->Username = 'stofinist@gmail.com';
                $mail->Password = 'Dkeuv82d1';
                $mail->CharSet =  'UTF-8'; // 'Windows-1251'

                $mail->SetFrom($mail->Username);
                $mail->AddAddress(trim('sto1s@finist.ru'));
                $mail->Subject = "Сделка для СТО {$this->postData['BitrixId']}";
                $mail->MsgHTML(" ");
                $mail->addAttachment($filename);

                if (!$mail->Send()) {
                    $this->jsonResponse['success'] = false;
                    $this->jsonResponse['message'] = "Не удалось отправить. Ошибка при отправке письма. $mail->ErrorInfo";
                } else {
                    $this->jsonResponse['success'] = true;
                }

                $mail->ClearAddresses();
                $mail->ClearAttachments();
            } catch (Exception $e) {
                $this->jsonResponse['success'] = false;
                $this->jsonResponse['message'] = "Не удалось отправить. Ошибка при отправке письма. $mail->ErrorInfo";
            }
        } else {
    if (!empty($this->dealData) && !empty($this->postData)) {
        $this->prepareSendPackage();

        if ($this->validate()) {
            $connect  = new \Gtd\Finist\UTP();
            $sendData = $this->sendData2Server;

            // ---- Департамент/руководитель (как было) ----
            $department = CIntranetUtils::GetUserDepartments($sendData['Responsible']['id']);
            if (is_array($department)) {
                $department_id = reset($department);
                $manager       = CIntranetUtils::GetDepartmentManager($department);
                $manager_id    = array_key_first($manager);

                $iblock = CIBlockSection::GetList(
                    [],
                    ["IBLOCK_ID" => 1, "ID" => $department_id],
                    false,
                    ['ID', 'NAME'],
                    false
                );

                if ($arElement = $iblock->Fetch()) {
                    $department_name = $arElement["NAME"];
                }

                $dep_info = [
                    'DEPARTMENT_ID'   => $department_id,
                    'DEPARTMENT_NAME' => $department_name ?? null,
                    'MANAGER_ID'      => $manager_id,
                ];

                $sendData = array_merge($sendData, $dep_info);
            }

            // ----------------- РЕФАКТОРИНГ ВАЛИДАЦИИ ПЕРЕД 1С -----------------
            $ourvoronka  = (int)($sendData["Category"] ?? 0);
            $truetype    = (int)($sendData["Type"] ?? 0);
            $ourmistake  = "Отправка в доставку не удалась";
            $showmistake = 0; // 0 = можно отправлять, 1 = нельзя отправлять

            // Вспомогательные функции (локально, чтобы просто вставить куском)
            $isEmptyField = function ($v): bool {
                return $v === null || $v === '';
            };

            $mbStriposSafe = function (string $haystack, string $needle) {
                if (function_exists('mb_stripos')) {
                    return mb_stripos($haystack, $needle, 0, 'UTF-8');
                }
                return stripos($haystack, $needle);
            };

            $getDealBpStates = function (int $dealId): array {
                $documentType = ["crm", "CCrmDocumentDeal", "DEAL"];
                $documentId   = ["crm", "CCrmDocumentDeal", 'DEAL_' . $dealId];
                $states       = CBPDocument::GetDocumentStates($documentType, $documentId);
                return is_array($states) ? $states : [];
            };

            $hasSuccessfulBp = function (array $states, array $rules): bool {
                foreach ($states as $st) {
                    $tplId = (int)($st['TEMPLATE_ID'] ?? 0);
                    $title = (string)($st['STATE_TITLE'] ?? '');

                    foreach ($rules as $r) {
                        if ($tplId !== (int)$r['template_id']) {
                            continue;
                        }
                        if (!isset($r['state_title'])) {
                            return true; // шаблон совпал, статус не важен
                        }
                        if ($title === (string)$r['state_title']) {
                            return true;
                        }
                    }
                }
                return false;
            };

            $detectProductFlags = function ($products) use ($mbStriposSafe): array {
                $flags = [
                    'kasko' => false,
                    'osago' => false,
                    'is'    => false,
                    'titul' => false,
                    'imush' => false,
                    'sk_banki' => false,
                ];

                if (!is_array($products)) {
                    return $flags;
                }

                foreach ($products as $p) {
                    $name = (string)($p['name'] ?? '');
                    if ($name === '') {
                        continue;
                    }

                    if (!$flags['kasko'] && $mbStriposSafe($name, 'КАСКО') !== false) {
                        $flags['kasko'] = true;
                    }
                    if (!$flags['osago'] && $mbStriposSafe($name, 'ОСАГО') !== false) {
                        $flags['osago'] = true;
                    }

                    if (!$flags['is'] && $name === 'ИС') {
                        $flags['is'] = true;
                    }
                    if (!$flags['titul'] && $name === 'ИС титул') {
                        $flags['titul'] = true;
                    }
                    if (!$flags['imush'] && $name === 'ИС имущество') {
                        $flags['imush'] = true;
                    }

                    if (!$flags['sk_banki'] && trim($name) === 'СК Банки.ру') {
                        $flags['sk_banki'] = true;
                    }

                    if (!in_array(false, $flags, true)) {
                        break;
                    }
                }

                return $flags;
            };
            $flags = $detectProductFlags($sendData['Product'] ?? []);
            $hasDealProductByName = function (string $expectedName): bool {
                $products = CCrmProductRow::LoadRows('D', $this->arParams['ID'], false);
                if (!is_array($products)) {
                    return false;
                }

                foreach ($products as $product) {
                    if (trim((string)($product['PRODUCT_NAME'] ?? '')) === $expectedName) {
                        return true;
                    }
                }

                return false;
            };
            $hasSkBankiProduct = $flags['sk_banki'] || $hasDealProductByName('СК Банки.ру');

            // Правила валидации только для (6 и 9) и только если type != 4
            if ($truetype != 4 && ($ourvoronka === 6 || $ourvoronka === 9)) {
                $dealId  = (int)($sendData["Deal"] ?? 0);
                $states  = $dealId > 0 ? $getDealBpStates($dealId) : [];
                $errors  = [];

                if ($ourvoronka === 6) {
                    // БП БСО 2.0
                    $bpOk = $hasSuccessfulBp($states, [
                        ['template_id' => 1259, 'state_title' => 'Внесение е-полисов: Бланки приняты на склад'],
                        ['template_id' => 1534, 'state_title' => 'Внесение е-полисов: Бланки приняты на склад'],
                    ]);

                    if (!$bpOk) {
                        $showmistake = 1;
                        $ourmistake  = "Данные по полису и выезду не отправлены в 1С, так как нет успешно закрытого БП “Бронирование БСО в 2.0”";
                    } else {
                        $showmistake = 0;
                    }

                    // Поля при наличии товаров (как у тебя: если ошибки есть — они приоритетнее текста про БП)
                    if ($flags['kasko'] && $isEmptyField($sendData['SNPolisKasko'] ?? null)) {
                        $showmistake = 1;
                        $errors[] = ":-(- Данные по полису и выезду не были отправлены в 1С, так как не заполнено поле Серия/номер полиса КАСКО или Серия\\Номер БСО КАСКО";
                    }
                    if ($flags['osago'] && $isEmptyField($sendData['SNPolisOsago'] ?? null)) {
                        $showmistake = 1;
                        $errors[] = "Данные по полису и выезду не были отправлены в 1С, так как не заполнено поле Серия/номер полиса ОСАГО";
                    }

                    if (!empty($errors)) {
                        $ourmistake = implode(', ', $errors);
                    }
                }

                if ($ourvoronka === 9) {
                    // БП Проект договора / пролонгация / ...
                    $bpOk = $hasSuccessfulBp($states, [
                        ['template_id' => 1461, 'state_title' => 'Бланки приняты на склад'],
                        ['template_id' => 1267], // без статуса, как у тебя
                        ['template_id' => 854,  'state_title' => 'БСО внесено на склад'],
                    ]);

                    if (!$bpOk) {
                        $showmistake = 1;
                        $ourmistake  = ":( Данные по полису и выезду не отправлены в 1С, так как нет успешно закрытого БП по принятию бланков: Проект договора КИС 1С или Пролонгация (не запускать) или ВЗР/СПОРТ Дети";
                    } else {
                        $showmistake = 0;
                    }

                    $baseMessage = "Данные по полису и выезду не были отправлены в 1С, так как не заполнено поле Серия/номер полиса";

                    if ($flags['is'] && $isEmptyField($sendData['SNPolisIS'] ?? null)) {
                        $showmistake = 1;
                        $errors[] = $baseMessage . " ИС";
                    }
                    if ($flags['titul'] && $isEmptyField($sendData['Titul'] ?? null)) {
                        $showmistake = 1;
                        $errors[] = $baseMessage . " ИС титул";
                    }
                    if ($flags['imush'] && $isEmptyField($sendData['Imushestvo'] ?? null)) {
                        $showmistake = 1;
                        $errors[] = $baseMessage . " ИС имущество";
                    }

                    if (!empty($errors)) {
                        $ourmistake = implode(', ', $errors);
                    }
                }
            }
            // ----------------- /РЕФАКТОРИНГ ВАЛИДАЦИИ -----------------

            // Отправка: только если не нашли ошибок
            if ($hasSkBankiProduct && $isEmptyField($this->dealData['UF_CRM_1536917510'] ?? null)) {
                $showmistake = 1;
                $ourmistake  = "СК банки товар поле не заполнено";
            }

            $response = [];
            if ($showmistake != 1) {
                $response = $connect->SendRestRequest('delivery', '', true, true, $sendData);
            } else {
                // Явно отдаём понятную ошибку, иначе дальше будет undefined $response
                $this->jsonResponse['success'] = false;
                $this->jsonResponse['message'] = $ourmistake;
                return;
            }

            if (!empty($response['guid'])) {
                $this->jsonResponse['success'] = true;
            } else {
                if (!empty($response["errorMessage"])) {
                    $ourmistake = $response["errorMessage"];
                }

                $this->jsonResponse['success'] = false;
                $this->jsonResponse['message'] = $ourmistake;
                Sentry\captureMessage('не удалось отправить в доставку');
            }
        } else {
            $this->jsonResponse['success'] = false;
            $this->jsonResponse['message'] = $this->getValidationMessage();
        }
    }
}

    }

    public function getPostBody()
    {
        return json_decode(file_get_contents("php://input"), true);
    }

    public function prepareSendPackage()
    {

        $this->sendPackegeMap['AgentId'] = [
            'FIELD' => $this->getAgentDealFieldName(),
            'FROM' => "DEAL",
            'TYPE' => "STRING"
        ];

        foreach ($this->sendPackegeMap as $field => $param) {
            $data = [];
            if ($param['FROM'] == 'POST')
                $data  = $this->postData;
            elseif ($param['FROM'] == 'DEAL')
                $data = $this->dealData;

            $val = $data[$param['FIELD']];
            if (!empty($param['FUNC'])) {
                if (method_exists($this, $param['FUNC'])) {
                    $method = $param['FUNC'];
                    $this->sendData2Server[$field] = $this->$method($val);
                }
            } else {
                if ($param['TYPE'] == 'STRING') {
                    $this->sendData2Server[$field] = $val;
                } elseif ($param['TYPE'] == 'ARRAY') {
                    $this->sendData2Server[$field] = $this->getEnumList($val);
                } elseif ($param['TYPE'] == "BOOLEN") {
                    $this->sendData2Server[$field] =  $val == 'Y' ? true : false;
                } elseif ($param['TYPE'] == 'ENUM') {
                    $this->sendData2Server[$field] = $this->getEnumList($val, false);
                }
            }
        }
    }

    /**
     * @param $user_id
     * @return array
     * @throws \Bitrix\Main\ArgumentException
     * @throws \Bitrix\Main\ObjectPropertyException
     * @throws \Bitrix\Main\SystemException
     */
    public function getClientInfo($user_id)
    {
        $arContactField = [];
        $contact = ContactTable::getList([
            'filter' => ['ID' => $user_id],
            'select' => ['*', 'UF_*']
        ])->fetch();
        $arContactField['name'] = $contact['LAST_NAME'] . ' ' . $contact['NAME'] . ' ' . $contact['SECOND_NAME'];
        $arContactField['contact'] = FieldMultiTable::getList([
            'filter' => [
                'ENTITY_ID' => 'CONTACT',
                'ELEMENT_ID' => $user_id,
                'TYPE_ID' => 'PHONE'
            ],
            'select' => ['VALUE', 'VALUE_TYPE']
        ])->fetchAll();
        return $arContactField;
    }

    public function getRespInfo($use_id)
    {
        $arUserInfo = [];
        if ($use_id > 0) {
            $uRes = CUser::GetList($by, $order, ['ID' => $use_id], ['SELECT' => ['*', 'UF_*']]);
            if ($arUser = $uRes->Fetch()) {
                $arUserInfo['id'] = $arUser['ID'];
                $arUserInfo['name'] = $arUser['LAST_NAME'] . ' ' . $arUser['NAME'];
                $arUserInfo['second_name'] = $arUser['SECOND_NAME'];
                $arUserInfo['phone'] = $arUser['UF_PHONE_INNER'];
                $arUserInfo['email'] = $arUser['EMAIL'];
                $arUserInfo['mobile_phone'] = $arUser['PERSONAL_MOBILE'];
                $arUserInfo['upravlenie'] = $arUser['UF_LINKEDIN'];
            }
        }
        return $arUserInfo;
    }

    public function getComment($val)
    {
        $contactStr = $this->getContactsInfoStr();
        return $val . "\n\r" . $contactStr;
    }

    public function getContactsInfoStr()
    {
        $arStringContact = [];

        $id = $this->dealData['ID'];
        if ($id > 0) {
            $arContactId = \Bitrix\Crm\Binding\DealContactTable::getDealContactIDs($id);
            foreach ($arContactId as $id) {
                $contactString = "";
                $contact = ContactTable::getByPrimary($id)->fetch();
                $contactString .= $contact['LAST_NAME'] . ' ' . $contact['NAME'] . ' ' . $contact['SECOND_NAME'];
                $contactString .= ': ';
                $contactString .= $this->getContactPhoneString($contact['ID']);
                $arStringContact[] = $contactString;
            }
        }
        return implode("\n\r", $arStringContact);
    }

    public function getContactPhoneString($id)
    {
        $phoneStr = '';
        $res = \Bitrix\Crm\FieldMultiTable::getList(
            [
                'filter' => [
                    'ENTITY_ID' => 'CONTACT',
                    'ELEMENT_ID' => $id,
                    'TYPE_ID' => 'PHONE'
                ], 'select' => ['VALUE']
            ]
        );
        while ($phone = $res->fetch()) {
            $phoneStr .= $phone['VALUE'] . ' ';
        }
        return $phoneStr;
    }

    public function getEnumList($arIds, $multiple = true)
    {
        $arFields = null;
        if (!empty($arIds)) {
            $res = CUserFieldEnum::GetList([], ['ID' => $arIds]);
            if ($multiple) {
                while ($field = $res->Fetch()) {
                    $arFields[] = $field['VALUE'];
                }
            } else {
                if ($field = $res->Fetch()) {
                    $arFields = $field['VALUE'];
                }
            }
        }
        return $arFields;
    }

    public function getProduct($val)
    {
        $arProduct = [];
        $arPostProduct = [];
        foreach ($this->postData['PRODUCT'] as $pPropd) {
            $arPostProduct[$pPropd['id']] = $pPropd;
        }
        $res = CCrmProductRow::LoadRows('D', $this->arParams['ID'], false);
        foreach ($res as  $k => $product) {
            //get off discount
            $prodProp = \Gtd\Finist\ProductPropertyTable::getList([
                'filter' => [
                    '=OWNER_ID' => $this->arParams['ID'],
                    '=PRODUCT_ID' => $product['PRODUCT_ID']
                ]
            ])->fetch();
            if ($arPostProduct[$product['ID']]['active']) {
                $arProduct[$k]['name'] = $product['PRODUCT_NAME'];
                $arProduct[$k]['price'] = $product['PRICE'];
                $arProduct[$k]['discount'] = $product['DISCOUNT_SUM'];
                $arProduct[$k]['payType'] = $arPostProduct[$product['ID']]['payType'];
                $arProduct[$k]['priceCustom'] = floatval($arPostProduct[$product['ID']]['price']);
                $arProduct[$k]['prolongation'] = $arPostProduct[$product['ID']]['prolongation'];
                $arProduct[$k]['Installment'] = $arPostProduct[$product['ID']]['installment'];
                $arProduct[$k]['surcharge'] = $arPostProduct[$product['ID']]['surcharge'];
                $arProduct[$k]['offDiscount'] = (int)$prodProp['OFF_DISCOUNT'];
                $arProduct[$k]['epolis'] = $arPostProduct[$product['ID']]['epolis'];
                $arProduct[$k]['longTerm'] = $arPostProduct[$product['ID']]['longTerm'];
            }
        }
        return $arProduct;
    }

    public function getMetro($val)
    {
        $metroClass = self::getHlClass(2);
        $res = $metroClass::getList(['filter' => ['=UF_XML_ID' => $val]])->fetch();
        return [
            'id' => $res['UF_XML_ID'],
            'name' => $res['UF_NAME']
        ];
    }

    public function getCity($val)
    {
        $cityClass = self::getHlClass(1);
        $res = $cityClass::getList(['filter' => ['ID' => $val]])->fetch();
        $City = [
            'id' => $res['UF_XML_ID'],
            'name' => $res['UF_NAME']
        ];
        return $City;
    }

    public function getDealData()
    {
        $deal = DealTable::getList(['filter' => ['ID' => $this->arParams['ID']], 'select' => ['*', 'UF_*']])->fetch();
        return $deal;
    }

    public function getPostData()
    {
        return $this->request->getPost('post');
    }

    public function showJsonResponce()
    {
        global $APPLICATION;
        $APPLICATION->RestartBuffer();
        header('Content-Type: application/json');
        echo json_encode($this->jsonResponse);
        die();
    }

    public function checkDocAvailability()
    {
        $deal = DealTable::getList(['filter' => ['ID' => $this->arParams['ID']], 'select' => ['*', 'UF_*']])->fetch();
        $dealCat = $deal['CATEGORY_ID'];
        $docDeal = !empty($deal['UF_CRM_1530183302']);

        //if (!!$deal['CONTACT_ID']) {
            $doc = DocumentTable::getList([
                'filter' => [
                    '=DEAL_CATEGORY_ID' => [$dealCat, 0],
                    '=CONTACT_ID' => $deal['CONTACT_ID']
                ]
            ])->fetch();
        //}

        $this->arResult['DocDealId'] = $this->arParams['ID'];
        $this->arResult['DocAvailability'] = false;
        if ($doc || $docDeal) {
            $this->arResult['DocAvailability'] = true;
        }
    }

    private function getSource($id)
    {
        $status = \Bitrix\Crm\StatusTable::getList([
            'filter' => [
                '=ENTITY_ID' => 'SOURCE',
                '=STATUS_ID' => $id
            ]
        ])->fetch();
        if ($status) {
            return $status['NAME'];
        }
        return "";
    }

    private function getChiefForUser($userId)
    {
        global $USER;

        $resUser = $USER->GetList($by, $order, ['ID' => $userId], ['SELECT' => ['ID', 'UF_DEPARTMENT']]);
        if ($user = $resUser->GetNext()) {
            $userDepartment = $user['UF_DEPARTMENT'][0];

			if ($userDepartment == 702) {$userDepartment = 54;}
            $res = CIBlockSection::GetNavChain(1, $userDepartment, ['NAME', 'ID', 'LEFT_MARGIN']);
            while ($sec = $res->GetNext()) {
                $result[] = $sec['ID'];
            }
        }
        if (!empty($result)) {
            $res = CIBlockSection::GetList(['LEFT_MARGIN' => 'DESC'], ['IBLOCK_ID' => 1, 'ID' => $result, '!UF_HEAD' => false], false, ['NAME', 'UF_HEAD']);
            while ($sec = $res->GetNext()) {
                if ($sec['UF_HEAD'] > 0) {
                    return $this->getRespInfo($sec['UF_HEAD']);
                }
            }
        }
        return false;
    }

    public function getAgentList()
    {
        return AgentTable::getList()->fetchAll();
    }

    public function loadLastDelivery()
    {
        $res = DeliveryTable::getList([
            'filter' => [
                'Deal_Id' => $this->arParams['ID']
            ],
            'order' => ['Number' => 'DESC']
        ])->fetch();
        if ($res && $res['Metro']) {
            $res['Date'] = $res['Date']->format('d/m/Y');
            $res['City'] = $this->getCityByMetroCode($res['Metro']);
            $this->jsonResponse['lastDelivery'] = $res;
        } else {
            $this->jsonResponse['lastDelivery'] = false;
        }
    }

    public function executeComponent()
    {

        if ($this->request->isPost()) {
            $this->ajaxPostData();
            $this->showJsonResponce();
        } elseif ($this->request->getQuery('loadProduct')  === 'Y') {
            $this->loadProduct();
            $this->jsonResponse['Product'] = $this->arResult['PRODUCT'];
            $this->loadLastDelivery();
            $this->showJsonResponce();
        } else {

            if ($this->arParams['ID']) {

                $this->arResult['deliveryType'] = $this->getTypeDelivery();
                $this->arResult['inspection'] = $this->getInspection();
                $this->arResult['city'] = $this->getMetroArray();
                //$this->arResult['agent'] = $this->getAgentList();

                $this->loadProduct();
                $this->checkDocAvailability();
            }

            $this->includeComponentTemplate();

        }
    }


    // VALIDATION
    private function validate()
    {
        $errorCount = 0;
        foreach ($this->validation as $rule) {
            $method = $rule['FUNC'];
            $field = $rule['FIELD'];
            $val = $this->sendData2Server[$field];
            if (method_exists($this, $method)) {
                if (!$this->$method($val)) {
                    $errorCount++;
                    $this->setValidationMessage($rule['MESSAGE']);
                }
            }
        }
        return $errorCount === 0;
    }

    /**
     * @param string $validatationMessage
     */
    public function setValidationMessage(string $validatationMessage): void
    {
        $this->validationMessage[] = $validatationMessage;
    }

    /**
     * @return string
     */
    public function getValidationMessage(): string
    {
        return implode('<br>', $this->validationMessage);
    }

    private function validateSource($val)
    {
        $validSource = ['Агентства недвижимости', 'Риелторы'];
        if (in_array($val, $validSource)) {
            return !empty($this->dealData['UF_CRM_5DC4FE904D612']);
        }
        return true;
    }
}

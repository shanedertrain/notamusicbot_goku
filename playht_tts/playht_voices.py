from enum import Enum


class StrEnum(str, Enum):
    """
    A backport of StrEnum for Python < 3.11.
    """

    def __str__(self) -> str:
        return str.__str__(self)


class VoiceManifest(StrEnum):
    Charlotte_Narrative = (
        "s3://mockingbird-prod/charlotte_vo_narrative_9290be17-ccea-4700-a7fd-a8fe5c49fb20/voices/speaker/manifest.json"
    )
    Susan_Advertising = (
        "s3://mockingbird-prod/susan_vo_commercials_0f4fa663-6eba-4582-be1e-2d5bde798f1c/voices/speaker/manifest.json"
    )
    Navya = "s3://voice-cloning-zero-shot/e5df2eb3-5153-40fa-9f6e-6e27bbb7a38e/original/manifest.json"
    Olivia_Advertising = (
        "s3://mockingbird-prod/olivia_vo_commercials_6e3c384f-15d6-4fe7-b9a4-0cb1d69daeba/voices/speaker/manifest.json"
    )
    Mason = "s3://voice-cloning-zero-shot/a540a448-a9ca-446c-9538-d1bae6c506f1/original/manifest.json"
    Indigo = "s3://voice-cloning-zero-shot/97580643-b568-4198-aaa4-3e07e4a06c47/original/manifest.json"
    Calvin = "s3://voice-cloning-zero-shot/743575eb-efdc-4c10-b185-a5018148822f/original/manifest.json"
    Amelia = "s3://voice-cloning-zero-shot/34eaa933-62cb-4e32-adb8-c1723ef85097/original/manifest.json"
    Sarge = "s3://mockingbird-prod/agent_47_carmelo_pampillonio_58e796e1-0b87-4f3e-8b36-7def6d65ce66/voices/speaker/manifest.json"
    William_Narrative = (
        "s3://mockingbird-prod/william_vo_narrative_0eacdff5-6243-4e26-8b3b-66e03458c1d1/voices/speaker/manifest.json"
    )
    Autumn = "s3://voice-cloning-zero-shot/ff414883-0e32-4a92-a688-d7875922120d/original/manifest.json"
    Isabella = "s3://voice-cloning-zero-shot/a0fa25cc-5f42-4dd0-8a78-a950dd5297cd/original/manifest.json"
    Arthur_Advertising = "s3://peregrine-voices/arthur ads parrot saad/manifest.json"
    Billy = "s3://mockingbird-prod/nathan_drake_carmelo_pampillonio_7d540ad6-7d32-41f6-8d53-2584901aa03d/voices/speaker/manifest.json"
    Richie = "s3://voice-cloning-zero-shot/dc90b58b-59a9-4e65-955d-c7620deb2d7a/original/manifest.json"
    Susan_Training = (
        "s3://mockingbird-prod/susan_vo_training_46ffcc60-d630-42f6-acfe-4affd003ae7a/voices/speaker/manifest.json"
    )
    Lachlan = "s3://voice-cloning-zero-shot/bb759cd0-edb0-43d9-8273-f0a7c048fb11/original/manifest.json"
    Atlas = "s3://voice-cloning-zero-shot/e46b4027-b38d-4d24-b292-38fbca2be0ef/original/manifest.json"
    Finley = "s3://voice-cloning-zero-shot/aa753d26-bc20-479f-95af-5c3c1c970d93/original/manifest.json"
    Pia = "s3://voice-cloning-zero-shot/74f59ea4-f07b-4d10-88d1-ca174adac3f3/original/manifest.json"
    Abigail = "s3://mockingbird-prod/abigail_vo_6661b91f-4012-44e3-ad12-589fbdee9948/voices/speaker/manifest.json"
    Nigel = "s3://voice-cloning-zero-shot/f8af54e1-1534-4f68-ad8a-260cc132f820/original/manifest.json"
    Aaliyah = "s3://voice-cloning-zero-shot/f6c4ed76-1b55-4cd9-8896-31f7535f6cdb/original/manifest.json"
    Arthur_Meditation = (
        "s3://mockingbird-prod/arthur_vo_meditatoin_211f702d-b185-4115-b8b4-801f8130a38d/voices/speaker/manifest.json"
    )
    Eileen = "s3://mockingbird-prod/eileen_vo_5d7b2bcc-d635-4301-97e8-d97c13768514/voices/speaker/manifest.json"
    Nolan = "s3://peregrine-voices/nolan saad parrot/manifest.json"
    Leroy = "s3://voice-cloning-zero-shot/32ae7ca0-634e-4fab-af74-0ec7c663e9da/original/manifest.json"
    William_Training = (
        "s3://mockingbird-prod/william_vo_training_1b939b71-14fa-41f0-b1db-7d94f194ad0a/voices/speaker/manifest.json"
    )
    Larry_Narrative = (
        "s3://mockingbird-prod/larry_vo_narrative_4bd5c1bd-f662-4a38-b5b9-76563f7b92ec/voices/speaker/manifest.json"
    )
    Ada = "s3://voice-cloning-zero-shot/72aafde0-8f4f-4a91-a483-21d76114ab17/original/manifest.json"
    Mitch = "s3://voice-cloning-zero-shot/8007f637-9aec-4426-8c6a-7e713c032623/original/manifest.json"
    Ayla_Narrative = (
        "s3://mockingbird-prod/ayla_vo_narrative_d8199dfd-b50f-40c7-9d99-e203ba5f4152/voices/speaker/manifest.json"
    )
    Lumi = "s3://voice-cloning-zero-shot/640a6636-dc16-4911-b75a-1549daae2c71/original/manifest.json"
    Hook = "s3://mockingbird-prod/hook_1_chico_a3e5e83f-08ae-4a9f-825c-7e48d32d2fd8/voices/speaker/manifest.json"
    Luna = "s3://voice-cloning-zero-shot/f43cc4b4-b193-4a13-a903-e6b125c3d572/original/manifest.json"
    Samara = "s3://voice-cloning-zero-shot/90217770-a480-4a91-b1ea-df00f4d4c29d/original/manifest.json"
    Baptiste = "s3://voice-cloning-zero-shot/1d26f4fe-1d08-4cfe-a7c1-d28e4e913ff9/original/manifest.json"
    Niamh = "s3://voice-cloning-zero-shot/928ed0a0-2271-4710-a7c9-1711d36b9897/original/manifest.json"
    Furio = "s3://voice-cloning-zero-shot/44f32760-8f41-4dfb-b192-ca077fc501ea/original/manifest.json"
    Alessandro = "s3://voice-cloning-zero-shot/7ced805f-611e-433c-8c43-568f48a8af4e/original/manifest.json"
    Archie = "s3://voice-cloning-zero-shot/ac9e2984-c7bb-44c8-8b6b-5c10728ad5cf/original/manifest.json"
    Waylon = "s3://voice-cloning-zero-shot/b4a876c1-8730-435e-9595-141799868808/original/manifest.json"
    Scarlett = "s3://voice-cloning-zero-shot/32b943f6-87cf-4e15-8e7a-d4cb848e3689/original/manifest.json"
    Teddy = "s3://voice-cloning-zero-shot/f3600d9d-39ab-4494-b6ac-c6f5d122184d/original/manifest.json"
    Alfonso = "s3://voice-cloning-zero-shot/5d243997-dcf6-4895-b504-eb5ce95a043e/alphonsosaad/manifest.json"
    Anthony = "s3://voice-cloning-zero-shot/b3def996-302e-486f-a234-172fa0279f0e/anthonysaad/manifest.json"
    Ariana = "s3://voice-cloning-zero-shot/f2863f63-5334-4f65-9d30-438feb79c2ec/arianasaad2/manifest.json"
    Arthur_Narrative = (
        "s3://voice-cloning-zero-shot/0326e8a4-9001-4d92-853b-3a14dd2ea38a/arthurnarrativesaad/manifest.json"
    )
    Arthur_Training = (
        "s3://voice-cloning-zero-shot/4bcdf603-fc5f-4040-a6dd-f8d0446bca9d/arthurtrainingsaad/manifest.json"
    )
    Ayla_Advertising = (
        "s3://voice-cloning-zero-shot/33e6b76a-7554-48fa-9798-2e6823ab0a10/aylaadvertisingsaad/manifest.json"
    )
    Ayla_Expressive = (
        "s3://mockingbird-prod/ayla_vo_expressive_16095e08-b9e8-429b-947c-47a75e41053b/voices/speaker/manifest.json"
    )
    Ayla_Meditation = (
        "s3://voice-cloning-zero-shot/f741f871-63ad-4207-8278-907aec4e9e50/aylameditationsaad/manifest.json"
    )
    Ayla_Training = (
        "s3://mockingbird-prod/ayla_vo_training_e6751ca5-e47c-4c4b-ad05-d3a194417600/voices/speaker/manifest.json"
    )
    Carmen = "s3://voice-cloning-zero-shot/fdb74aec-ede9-45f8-ad87-71cb45f01816/original/manifest.json"
    Barry_Advertising = "s3://peregrine-voices/barry ads parrot saad/manifest.json"
    Barry_Narrative = "s3://peregrine-voices/barry narrative parrot saad/manifest.json"
    Susan_Narrative = (
        "s3://mockingbird-prod/susan_vo_narrative_73051c90-460b-4e54-adab-9235f45c5e5f/voices/speaker/manifest.json"
    )
    Charlotte_Advertising = "s3://peregrine-voices/charlotte ads parrot saad/manifest.json"
    Charlotte_Meditation = "s3://peregrine-voices/charlotte meditation 2 parrot saad/manifest.json"
    Charlotte_Training = "s3://peregrine-voices/charlotte_training_parrot_saad/manifest.json"
    Chris = "s3://voice-cloning-zero-shot/028a32d4-6a79-4ca3-a303-d6559843114b/chris/manifest.json"
    Chuck = "s3://voice-cloning-zero-shot/40738a3a-34bb-4ac3-97c5-aed7b31ccf1d/chucksaad/manifest.json"
    Adelaide = "s3://voice-cloning-zero-shot/f9bf96ae-19ef-491f-ae69-644448800566/original/manifest.json"
    Olivia_Training = (
        "s3://mockingbird-prod/olivia_vo_training_4376204f-a411-4e5d-a5c0-ce6cc3908052/voices/speaker/manifest.json"
    )
    Davis = "s3://peregrine-voices/a10/manifest.json"
    Sumita = "s3://voice-cloning-zero-shot/f3c22a65-87e8-441f-aea5-10a1c201e522/original/manifest.json"
    Donna_Meditation = "s3://peregrine-voices/donna_meditation_saad/manifest.json"
    Donna_Narrative = "s3://peregrine-voices/donna_parrot_saad/manifest.json"
    Siobhán = "s3://voice-cloning-zero-shot/30884451-1eff-4fd8-9a24-d1ee3353b215/original/manifest.json"
    Evelyn = "s3://peregrine-voices/evelyn 2 saad parrot/manifest.json"
    Ranger = "s3://voice-cloning-zero-shot/abc2d0e6-9433-4dcc-b416-0b035169f37e/original/manifest.json"
    Oliver_Training = (
        "s3://mockingbird-prod/oliver_vo_training_6e3f604a-5605-4542-948d-347b0d7546fc/voices/speaker/manifest.json"
    )
    Fletcher = "fletcher"
    George = "s3://voice-cloning-zero-shot/418a94fa-2395-4487-81d8-22daf107781f/george/manifest.json"
    Hudson = "s3://peregrine-voices/hudson saad parrot/manifest.json"
    Jack = "s3://peregrine-voices/mel28/manifest.json"
    Jennifer = "s3://voice-cloning-zero-shot/801a663f-efd0-4254-98d0-5c175514c3e8/jennifer/manifest.json"
    Joseph = "s3://voice-cloning-zero-shot/dc23bb38-f568-4323-b6fb-7d64f685b97a/joseph/manifest.json"
    Larry_Advertising = "s3://peregrine-voices/larry_ads3_parrot_saad/manifest.json"
    Mark = "s3://voice-cloning-zero-shot/0b5b2e4b-5103-425e-8aa0-510dd35226e2/mark/manifest.json"
    Matt = "s3://voice-cloning-zero-shot/09b5c0cc-a8f4-4450-aaab-3657b9965d0b/podcaster/manifest.json"
    Melissa = "s3://peregrine-voices/mel21/manifest.json"
    Michael = "s3://voice-cloning-zero-shot/7c339a9d-370f-4643-adf5-4134e3ec9886/mlae02/manifest.json"
    Nicole = "s3://voice-cloning-zero-shot/7c38b588-14e8-42b9-bacd-e03d1d673c3c/nicole/manifest.json"
    Oliver_Advertising = "s3://peregrine-voices/oliver_ads2_parrot_saad/manifest.json"
    Oliver_Narrative = "s3://peregrine-voices/oliver_narrative2_parrot_saad/manifest.json"
    Olivia_Narrative = "s3://peregrine-voices/olivia_ads3_parrot_saad/manifest.json"
    Ruby = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json"
    Russell = "s3://peregrine-voices/russell2_parrot_saad/manifest.json"
    Samuel = "s3://voice-cloning-zero-shot/36e9c53d-ca4e-4815-b5ed-9732be3839b4/samuelsaad/manifest.json"
    Sarah = "s3://voice-cloning-zero-shot/820da3d2-3a3b-42e7-844d-e68db835a206/sarah/manifest.json"
    Siobhan = "s3://voice-cloning-zero-shot/30884451-1eff-4fd8-9a24-d1ee3353b215/original/manifest.json"
    Sophia = "s3://voice-cloning-zero-shot/1f44b3e7-22ea-4c2e-87d0-b4d9c8f1d47d/sophia/manifest.json"
    Will = "s3://peregrine-voices/mel22/manifest.json"


# Example Usage
if __name__ == "__main__":
    # Accessing enum members
    print(VoiceManifest.Charlotte_Narrative.value)
    print(VoiceManifest.Oliver_Training.value)

    # Iterating through all members
    for voice in VoiceManifest:
        print(f"{voice.name}: {voice.value}")
